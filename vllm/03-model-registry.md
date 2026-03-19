# vLLM 多模型支持与注册

## 核心定义（What & Why）

> vLLM 用**模型注册表（ModelRegistry）**把"HuggingFace 架构名字符串"映射到 vLLM 实现类，解耦模型发现与模型实现，使得支持新架构只需：① 实现符合协议的模型类，② 注册一行映射——引擎代码无需改动。

核心要点：

- `ModelRegistry` 是一个全局字典 `{arch_name: (module_path, class_name)}`
- 模型类必须实现 `load_weights()` 和标准 `forward()` 签名
- 多模态模型通过继承 `SupportsMultiModal` 协议扩展
- 社区贡献模型通过 `--trust-remote-code` 或插件机制动态注册

---

## ModelRegistry 工作原理

```python
# vllm/model_executor/models/__init__.py（简化）
_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM":      ("vllm.model_executor.models.llama",   "LlamaForCausalLM"),
    "MistralForCausalLM":    ("vllm.model_executor.models.mistral", "MistralForCausalLM"),
    "Qwen2ForCausalLM":      ("vllm.model_executor.models.qwen2",   "Qwen2ForCausalLM"),
    "Gemma2ForCausalLM":     ("vllm.model_executor.models.gemma2",  "Gemma2ForCausalLM"),
    "MixtralForCausalLM":    ("vllm.model_executor.models.mixtral", "MixtralForCausalLM"),
    # ... 50+ 架构
}

class ModelRegistry:
    @classmethod
    def resolve_model_cls(cls, architectures: List[str]):
        for arch in architectures:
            if arch in _MODELS:
                module_path, class_name = _MODELS[arch]
                module = importlib.import_module(module_path)  # 延迟导入
                return getattr(module, class_name)
        raise ValueError(f"架构 {architectures} 未注册")
```

`architectures` 来自 HuggingFace `config.json` 的 `"architectures"` 字段，vLLM 取第一个匹配项。

---

## 模型类必须实现的接口

```python
class MyModelForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # 用 vllm 的并行层（ColumnParallelLinear 等）搭建模型

    def forward(
        self,
        input_ids: torch.Tensor,           # [num_tokens]  (连续批处理，无 batch dim)
        positions: torch.Tensor,            # [num_tokens]  用于 RoPE
        kv_caches: List[torch.Tensor],      # 每层一个 KV cache tensor
        attn_metadata: AttentionMetadata,   # prefill/decode 的 slot mapping 等
    ) -> torch.Tensor:                      # 返回 logits [num_tokens, vocab_size]
        ...

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 将权重名映射到模型参数并原地赋值
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # 处理名称映射（HF → vLLM）
            name = self._rename_weight(name)
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # weight_loader 由 TP 并行层（如 ColumnParallelLinear）在初始化时挂载，
                # 负责将加载进来的完整权重按当前 tp_rank 切片后写入 param
                weight_loader(param, loaded_weight)
```

**关键约束**：

| 约束 | 原因 |
|------|------|
| `forward()` 无 batch 维度 | 连续批处理将所有 token 拼接成一维向量 |
| `input_ids` 为 `[num_tokens]` | prefill + decode token 混合在同一 tensor |
| 必须接受 `attn_metadata` | 注意力计算需要 slot mapping、block tables |
| 使用 vLLM 并行层 | 否则 TP 切分的 weight shard 与标准 `nn.Linear` 不兼容 |

---

## 当前支持的主要架构

| 类别 | 典型架构 |
|------|----------|
| **LLaMA 系** | LlamaForCausalLM, Llama3, CodeLlama, Vicuna |
| **Mistral/Mixtral** | MistralForCausalLM, MixtralForCausalLM（MoE）|
| **Qwen 系** | Qwen2ForCausalLM, Qwen2MoeForCausalLM, QwenForCausalLM |
| **Gemma 系** | GemmaForCausalLM, Gemma2ForCausalLM |
| **Phi 系** | PhiForCausalLM, Phi3ForCausalLM, Phi3SmallForCausalLM |
| **DeepSeek 系** | DeepseekV2ForCausalLM, DeepseekV3ForCausalLM（MLA + MoE）|
| **GPT 系** | GPT2LMHeadModel, GPTNeoXForCausalLM, GPTJForCausalLM |
| **多模态** | LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration |

---

## 如何接入新模型（完整步骤）

```
1. 创建模型文件
   vllm/model_executor/models/my_model.py
   └── 实现 MyModelForCausalLM（继承 nn.Module）

2. 注册到模型表
   vllm/model_executor/models/__init__.py
   _MODELS["MyModelForCausalLM"] = (
       "vllm.model_executor.models.my_model",
       "MyModelForCausalLM"
   )

3. 处理权重名映射
   HuggingFace 权重名（如 model.layers.0.self_attn.q_proj.weight）
   可能与 vLLM 实现不一致，在 load_weights() 中做重命名

4. 适配并行层
   - 用 ColumnParallelLinear 替换 QKV / FFN gate+up
   - 用 RowParallelLinear 替换 O / FFN down
   - 用 VocabParallelEmbedding 替换 embedding + lm_head

5. 使用社区插件方式（无需 fork）
   LLM(model="...", trust_remote_code=True)
   # 或通过 VLLM_WORKER_MULTIPROC_METHOD 和自定义 Python 包注册
```

---

## 多模态模型的额外接口

```python
class MyVisionLanguageModel(nn.Module, SupportsMultiModal):

    def get_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        **mm_kwargs,          # pixel_values, image_sizes, ...
    ) -> Optional[NestedTensors]:
        # 返回视觉 token 的 embedding，替换 input_ids 对应位置
        ...

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        # 将文本 embedding 和视觉 embedding 融合
        ...
```

---

## 关联知识网络

**前置**：[模型加载机制](02-model-loading.md)、[MoE 最小知识](../ai-infra/03-llm-architecture/06-moe-minimum.md)

**平行**：[算子替换与定制](04-custom-ops-and-operator-replacement.md)

**延伸**：[推理加速技术](05-inference-acceleration.md)、[Transformer 最小知识](../ai-infra/03-llm-architecture/01-transformer-minimum.md)

---

## 💥 实战踩坑记录

**现象**：用 `trust_remote_code=True` 加载社区模型，出现 `AttributeError: 'xxx' object has no attribute 'attn_metadata'`。

**根因**：社区模型的 `forward()` 签名遵循 HuggingFace 标准（有 `attention_mask`、`past_key_values`），而 vLLM 要求无这些参数、改用 `attn_metadata`。

**修复**：在社区模型中添加 vLLM 适配 wrapper，或者向 vLLM 提交 PR 实现原生支持。

---

**现象**：新增模型加载后，TP=4 下某些权重 shape 不匹配导致 crash。

**根因**：QKV 权重的切分方式与 multi-head / grouped-query head 数有关。GQA 模型中 K/V 的 head 数少于 Q，切分逻辑需要单独处理（`QKVParallelLinear` 内部有特殊路径）。

**修复**：使用 `QKVParallelLinear` 而非手动三个 `ColumnParallelLinear`，它内置了 GQA 兼容的切分逻辑。

---

## 🎯 面试高频 Q&A

**Q1：vLLM 如何自动找到应该加载哪个模型类？**

> 解析 HuggingFace `config.json` 的 `architectures` 字段，取第一个存在于 `ModelRegistry._MODELS` 字典中的名称，通过 `importlib.import_module` 懒加载对应模块并返回类。整个过程通过 `ModelConfig.verify_with_parallel_config()` 在引擎初始化时完成。

**Q2：支持新架构为什么必须使用 vLLM 自己的并行层，不能直接用 `nn.Linear`？**

> vLLM 的并行层（`ColumnParallelLinear` 等）在 `load_weights()` 时会自动按 `tp_rank` 和 `tp_size` 切片，并在 `forward()` 中插入必要的 all-reduce / all-gather 通信。普通 `nn.Linear` 无法感知分片逻辑，TP > 1 时结果错误且无 grad 通信。

**Q3：DeepSeek 的 MLA 注意力在 vLLM 中是怎么实现的，和标准 MHA 有何不同？**

> MLA 使用低秩 KV 压缩，KV cache 存储的是压缩后的 latent 向量而非解压后的 K/V。vLLM 为 DeepSeek 实现了专用的 `MLAAttention` 层，在 `attn_metadata` 中增加了解压矩阵 cache，并在 attention backend 中实现了 absorb-W 优化（将解压矩阵吸收进 QK 矩阵，避免运行时展开）。

---

[← 模型加载机制](02-model-loading.md) | [返回 vLLM 索引](00-index.md) | [下一篇：算子替换与定制 →](04-custom-ops-and-operator-replacement.md)
