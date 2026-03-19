# vLLM 模型加载机制

## 核心定义（What & Why）

> vLLM 的模型加载不是简单的 `torch.load()`，而是一条**配置解析 → 架构注册查找 → 分片权重流式加载 → Tensor Parallel 分布 → KV Cache 预分配**的完整流水线，核心目标是让任意 HuggingFace 兼容模型能以最低显存开销安全上线。

核心要点：

- `ModelConfig` 读取 `config.json`，确定架构名称和精度
- `ModelRegistry` 将架构名映射到 vLLM 实现类
- 权重通过 `BaseModelLoader` 流式加载，**不在 CPU 全量拷贝**
- Tensor Parallel 切分在加载时完成，每卡只持有自己的分片
- KV Cache 在模型加载后**单独预分配**，与模型权重物理隔离

---

## 模型加载全路径

```
EngineArgs.create_engine_config()
    └── ModelConfig(model, dtype, max_model_len, ...)
            ├── 读取 config.json → hf_config
            ├── 确定 architectures[0] = "LlamaForCausalLM"
            └── 推断 dtype（auto / float16 / bfloat16）

Worker.init_model()
    └── get_model(model_config, device_config, ...)
            ├── ModelRegistry.resolve_model_cls("LlamaForCausalLM")
            │       → vllm.model_executor.models.llama.LlamaForCausalLM
            ├── model = cls(config, cache_config, quant_config)
            │       → 在 meta device 上实例化（不分配实际显存）
            └── loader.load_model(model, ...)
                    ├── 打开权重文件（safetensors / pytorch_bin）
                    ├── 逐 tensor 流式读取
                    ├── 按 TP rank 筛选/切分该卡应持有的分片
                    └── model.load_weights(weights_iterator)
                            → 将切片 copy 到 GPU HBM

Worker.init_cache()
    └── CacheEngine(cache_config, model_config, parallel_config)
            ├── 计算可用显存 = total - model_weights - reserved
            ├── 确定 num_gpu_blocks = free_mem / block_size
            └── 分配 kv_caches: List[Tensor]  # shape: [num_blocks, ...]
```

---

## 关键配置参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `dtype` | `auto` | 权重精度；auto 时从 config.json 推断 |
| `gpu_memory_utilization` | `0.9` | 允许用于 KV cache 的显存比例上限 |
| `max_model_len` | 模型最大值 | 限制序列长度，影响每个 block 的 slot 数 |
| `tensor_parallel_size` | `1` | TP 维度，决定每卡持有的权重切片大小 |
| `load_format` | `auto` | 权重格式：`safetensors` / `pt` / `npcache` / `dummy` |
| `quantization` | `None` | 量化方案（awq / gptq / squeezellm / marlin）|

---

## 权重加载器的选择

```python
# vllm/model_executor/model_loader/loader.py
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)      # 测试用，随机初始化
    if load_config.load_format == LoadFormat.TENSORIZER:
        return TensorizerLoader(load_config)      # CoreWeave Tensorizer 加速
    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)    # 预分片存储，直接 mmap
    ...
    return DefaultModelLoader(load_config)        # 默认：safetensors / pt
```

`DefaultModelLoader` 的核心逻辑：

```python
def load_model(self, model, model_config, ...):
    # 1. 在 meta device 上构建模型（0 显存）
    with torch.device("meta"):
        model = model_class(hf_config, ...)

    # 2. 流式遍历权重文件
    for name, param in self._get_weights_iterator(model_path):
        # 3. 按 TP rank 切分
        shard = self._maybe_shard(name, param, tp_rank, tp_size)
        # 4. 原地填充到 meta tensor（materialize）
        model.load_weights([(name, shard)])
    return model.cuda()
```

---

## Tensor Parallel 权重切分规则

vLLM 在 `vllm/model_executor/layers/linear.py` 中将线性层分为三类：

| 类型 | 切分维度 | 代表层 |
|------|----------|--------|
| `ColumnParallelLinear` | 按列切（输出维度） | QKV projection, FFN gate/up |
| `RowParallelLinear` | 按行切（输入维度） | O projection, FFN down |
| `VocabParallelEmbedding` | 按词表切 | token embedding, lm_head |

关键：**每卡加载时只读取属于自己 rank 的分片**，不需要先全量加载再通信广播，这大幅降低了 CPU 内存峰值。

---

## KV Cache 显存估算

```
num_gpu_blocks = floor(
    (total_gpu_mem × gpu_memory_utilization - model_weight_mem)
    / (block_size × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes)
)
```

- `block_size`：默认 16（tokens / block）
- `× 2`：K + V 两个矩阵
- `dtype_bytes`：FP16=2, BF16=2, FP8=1

**实例**：LLaMA-3-8B 在单张 A100 80GB 上：

- 模型权重（BF16）≈ 16 GB
- 可用于 KV cache ≈ 80 × 0.9 − 16 = 56 GB
- 每 block（16 tokens）≈ 32 layers × 8 heads × 128 dim × 2 × 2B ≈ 1 MB
- num_gpu_blocks ≈ 56,000 个 block ≈ 896,000 个 token slot

---

## 关联知识网络

**前置**：[vLLM 架构总览](01-architecture-overview.md)、[Transformer 最小知识](../03-llm-architecture/01-transformer-minimum.md)

**平行**：[多模型支持与注册](03-model-registry.md)、[算子替换与定制](04-custom-ops-and-operator-replacement.md)

**延伸**：[推理加速技术](05-inference-acceleration.md)、[训练资源核算](../03-llm-architecture/07-training-resource-accounting.md)

---

## 💥 实战踩坑记录

**现象 1**：启动时报 `OutOfMemoryError`，但 `nvidia-smi` 显示显存充足。

**根因**：`gpu_memory_utilization=0.9` 是在模型权重加载**之后**计算剩余显存的，如果模型权重本身超过 `total × 0.9`，KV cache 预分配就会失败。

**修复**：降低 `gpu_memory_utilization` 或启用 TP（`tensor_parallel_size=2`）分摊权重。

---

**现象 2**：`safetensors` 加载比 `torch.load` 慢很多。

**根因**：部分环境的 `safetensors` 没有开启 `mmap` 模式，退化为全量 read。可通过 `load_format=npcache` 先转换为 numpy 内存映射格式，后续加载几乎零拷贝。

---

## 🎯 面试高频 Q&A

**Q1：vLLM 如何在不全量加载权重到 CPU 的情况下完成 TP 切分？**

> vLLM 在 `meta` device 上实例化模型（只构图，不分配内存），然后流式遍历权重文件，每读一个 tensor 立即按当前 rank 切分并写入 GPU，CPU 上同一时刻最多只缓存一层的权重。这使得 CPU 内存峰值从 `2 × model_size` 降至约 `1 layer_size`。

**Q2：`gpu_memory_utilization` 设为 1.0 是否安全？**

> 不安全。vLLM 用一次 dummy forward（输入填满 `max_model_len`）来测量模型激活值的显存占用，再用 `gpu_memory_utilization` 乘以剩余显存得到 KV cache 预算。留出 10% 余量是为了防止 CUDA context、临时 buffer、碎片等额外开销导致 OOM。

**Q3：模型权重文件格式 safetensors vs pytorch_bin 有何本质区别？**

> `safetensors` 头部存储了每个 tensor 的 byte offset 和 dtype，支持随机访问（mmap）和懒加载，因此可以按需只读取 TP rank 所需的列/行；`pytorch_bin`（pickle 格式）必须顺序反序列化整个文件，无法部分读取，TP 场景下 CPU 内存峰值更高。

---

[← 架构总览](01-architecture-overview.md) | [返回 vLLM 索引](00-index.md) | [下一篇：多模型支持与注册 →](03-model-registry.md)
