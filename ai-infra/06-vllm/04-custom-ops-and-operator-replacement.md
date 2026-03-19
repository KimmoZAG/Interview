# vLLM 算子替换与定制

## 核心定义（What & Why）

> vLLM 把"用哪个 kernel 计算注意力"和"用哪种方式执行线性层"从模型结构中**解耦**，通过 `AttentionBackend` 接口和量化插件机制实现热插拔，使同一份模型代码可以在 FlashAttention、FlashInfer、xFormers 等不同内核之间切换，也可以在不改动 forward 逻辑的前提下注入 INT8/INT4 量化。

核心要点：

- **注意力后端**：通过 `AttentionBackend` 抽象接口切换 kernel，与模型代码无关
- **量化替换**：`QuantizationConfig` 驱动，将 `nn.Linear` 替换为量化线性层
- **自定义 CUDA 核**：RoPE、RMSNorm、激活函数等用 Triton 或 CUDA C++ 实现
- 切换发生在**模型实例化阶段**，不影响 forward 函数签名

---

## 注意力后端系统

### 接口定义

```python
# vllm/attention/backends/abstract.py
class AttentionBackend(ABC):

    @staticmethod
    @abstractmethod
    def get_name() -> str: ...

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]: ...

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]: ...

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]: ...

class AttentionImpl(ABC):
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,      # [num_tokens, num_heads, head_size]
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,   # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: AttentionMetadata,
        ...
    ) -> torch.Tensor: ...
```

### 后端选择逻辑

```python
# vllm/attention/__init__.py
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: str,
    block_size: int,
) -> Type[AttentionBackend]:

    # 1. 用户强制指定 VLLM_ATTENTION_BACKEND 环境变量
    backend_name = envs.VLLM_ATTENTION_BACKEND
    if backend_name:
        return resolve_by_name(backend_name)

    # 2. 自动选择
    if is_hip():                        # AMD GPU → ROCm Flash
        return ROCmFlashAttentionBackend
    if head_size not in (64, 80, 96, 128, 256):
        return XFormersBackend          # 非标准 head size
    if dtype == torch.float32:
        return XFormersBackend          # FA 不支持 fp32
    return FlashAttentionBackend        # 默认：NVIDIA + fp16/bf16
```

### 各后端对比

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| `FlashAttentionBackend` | NVIDIA A10G / A100 / H100，fp16/bf16 | 最高吞吐；IO-aware tiling |
| `FlashInferBackend` | 需要 per-request 可变 KV layout | 支持 ragged KV，适合 prefix cache |
| `XFormersBackend` | 非标准 head size / fp32 | 兼容性好，性能略低 |
| `ROCmFlashAttentionBackend` | AMD GPU（MI200/MI300）| HIP 编译，接口与 FA 对齐 |
| `PlaceholderAttentionBackend` | CPU / Neuron 推理 | 纯 PyTorch 实现，无 CUDA |

---

## 量化线性层替换

### 插入时机

量化替换在 `get_model()` 阶段发生，通过 `quant_config` 改变 `Linear` 层的实例化：

```python
# 正常路径
linear = ColumnParallelLinear(in_features, out_features, ...)

# 量化路径（quant_config 非 None 时）
linear = quant_config.get_quant_method(layer).create_weights(
    layer, input_size, output_size, ...
)
```

### 支持的量化方案

| 方案 | 精度 | kernel | 适用场景 |
|------|------|--------|----------|
| `AWQ` | INT4 | AWQ CUDA / Marlin | 低 VRAM 部署，4-bit 权重 |
| `GPTQ` | INT4 / INT3 / INT2 | GPTQ-CUDA / Marlin | HuggingFace 量化模型 |
| `Marlin` | INT4 + FP16 | Marlin sparse GEMM | A100/H100 高吞吐 |
| `SqueezeLLM` | INT4（稀疏）| SqueezeLLM CUDA | 稀疏量化研究 |
| `FP8` | FP8-E4M3 | cuBLAS FP8 GEMM | H100 FP8 原生支持 |
| `BitsAndBytes` | INT8 / INT4 NF4 | bitsandbytes | 兼容 HF bnb 量化模型 |

### AWQ 替换示例

```python
# 原始：标准线性层
class LlamaAttention(nn.Module):
    self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim)

# AWQ 量化后：被替换为量化线性层
# vllm/model_executor/layers/quantization/awq.py
class AWQLinearMethod(LinearMethodBase):
    def create_weights(self, layer, ...):
        # 注册量化权重 (qweight, qzeros, scales) 而非 float weight
        layer.register_parameter("qweight", ...)
        layer.register_parameter("qzeros",  ...)
        layer.register_parameter("scales",  ...)

    def apply(self, layer, x, bias=None):
        # 调用 AWQ CUDA kernel 完成 dequant + GEMM
        return awq_ext.gemm_forward_cuda(x, layer.qweight, layer.scales, layer.qzeros)
```

---

## 自定义 CUDA 核（Custom Ops）

vLLM 用 Triton 和 CUDA C++ 实现了多个性能关键算子：

| 算子 | 实现 | 作用 |
|------|------|------|
| `RoPE` | CUDA C++ | Rotary Position Embedding，原地修改 Q/K |
| `RMSNorm` | Triton + CUDA | 融合 norm + residual add |
| `SiLU × Gate (SwiGLU)` | Triton | FFN 激活，融合两路相乘 |
| `Paged Attention v1/v2` | CUDA C++ | vLLM 原创，支持 block table 的注意力 |
| `AWQ / GPTQ GEMM` | CUDA C++ | INT4 反量化矩阵乘 |
| `Allreduce` | CUDA C++ | TP 内 custom fast allreduce（绕过 NCCL）|

自定义核的注册方式：

```python
# vllm/_custom_ops.py（PyTorch custom_op 机制）
import torch
import vllm._C  # 编译好的 CUDA 扩展

# 直接调用
vllm._C.ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
# is_neox: 是否使用 GPT-NeoX 风格的 RoPE（交替维度排列，区别于 LLaMA 风格的前后折叠）

# 或通过 ops 模块统一分发
from vllm import _custom_ops as ops
ops.rms_norm(out, hidden_states, weight, epsilon)
```

---

## 新后端 / 算子接入指南

```
接入新注意力后端：
1. 继承 AttentionBackend，实现 4 个静态方法
2. 继承 AttentionImpl，实现 forward()
3. 继承 AttentionMetadata，添加后端特有字段
4. 在 get_attn_backend() 中添加选择条件

接入新量化方案：
1. 继承 QuantizationConfig，解析量化 config
2. 继承 LinearMethodBase，实现 create_weights() + apply()
3. 在 _QUANTIZATION_CONFIG_REGISTRY 中注册名称
4. 在模型 load_weights() 中正确加载量化参数

替换单个算子（如自定义 RoPE）：
1. 注册 CUDA 或 Triton kernel（torch.library 或编译为 .so）
2. 在对应模型类中替换调用点
   - 直接替换：model 层内直接调用新 kernel
   - 后端注入：通过 AttentionImpl 的 forward 钩子传入
```

---

## 关联知识网络

**前置**：[模型加载机制](02-model-loading.md)、[FlashAttention IO-aware](../01-operator-optimization/06-flashattention-io-aware.md)、[量化 Basics](../01-operator-optimization/05-quantization-basics.md)

**平行**：[多模型支持与注册](03-model-registry.md)

**延伸**：[推理加速技术](05-inference-acceleration.md)、[kernel 执行模型](../01-operator-optimization/02-kernel-execution-model.md)

---

## 💥 实战踩坑记录

**现象**：切换到 `FlashInferBackend` 后，长序列请求偶发 CUDA illegal memory access。

**根因**：FlashInfer 的 `paged_prefill_with_kvcache` 要求 block table 在 GPU 上连续，但 BlockSpaceManager 默认允许 CPU 侧的非连续映射在最后一刻同步。

**修复**：在 `ModelRunner.prepare_model_input()` 中确保 `block_tables` 在传入 FlashInfer 前已完成 `contiguous()` + `to(device)` 操作。

---

**现象**：使用 AWQ INT4 模型，TP=2 下权重加载后 inference 结果全错。

**根因**：AWQ 的 `qweight`（INT32 pack 格式）按列切分时需要对齐 pack group（每 8 个 INT4 打包成 1 个 INT32），直接按 `tp_size` 整除切可能破坏 pack 边界。

**修复**：使用 vLLM 内置的 `AWQLinearMethod.load_weights()`，它内部处理了 pack-aware 切分逻辑。

---

## 🎯 面试高频 Q&A

**Q1：vLLM 中 FlashAttention 和 PagedAttention 是什么关系？**

> **FlashAttention** 是注意力的 IO-aware 计算算法，核心优化是分块 tiling 以减少 HBM 读写。**PagedAttention** 是 KV cache 的**内存管理机制**，将 KV 存储在非连续物理 block 中。两者正交：vLLM 的 `FlashAttentionBackend` 使用的 kernel 实现了**同时支持 block table + IO-aware tiling 的注意力算子**，可以理解为 PagedAttention 机制 + FlashAttention 算法的结合体。

**Q2：为什么同一个模型在 H100 和 A10G 上可能用不同的注意力 backend？**

> `get_attn_backend()` 根据 GPU 型号、dtype、head_size 自动选择。H100 支持 FP8 和更大的 SRAM（50 MB L2），可以充分利用 FlashAttentionV3 的新特性；A10G SRAM 较小（24 MB），在极长序列下 FA 的 tiling 块更小，FlashInfer 的 split-k 策略有时更优。实际部署建议 benchmark 两者。

**Q3：如何验证量化替换后数值精度没有严重下降？**

> vLLM 提供 `--dtype auto --quantization awq` 等参数后，可通过以下方式验证：① 与 FP16 基线对比同一 prompt 的 logits（用 `--max-tokens 1` 截断，比较 top-5 token 和概率）；② 运行 lm-evaluation-harness 测试 MMLU/HellaSwag 分数；③ 检查激活值是否出现 NaN（`VLLM_FULL_STOP_ON_NAN=1`）。INT4 AWQ 典型降分 < 1 个点，若超过 2 点需检查 group_size 配置。

---

[← 多模型支持](03-model-registry.md) | [返回 vLLM 索引](00-index.md) | [下一篇：推理加速技术 →](05-inference-acceleration.md)
