# AMD 上也能 True On-Policy 了，Triton Backend 全链路 Bitwise 对齐

Zhiyao Jiang & Xinyu Jiang
RL & LLM @ SGLang & Slime

---

## Power Up True On-Policy Training on AMD GPUs


## True On-Policy 为什么重要

On-Policy RL 训练中，有一个容易被忽略但至关重要的问题：**推理引擎（SGLang）生成 rollout 时计算的 log probability，和训练引擎（FSDP/Megatron）forward 时重新计算的 log probability，是否严格一致？**

在标准流程中，这两者之间总存在微小的数值差异——不同的 attention kernel、不同的 batch size、不同的浮点累积顺序，都会引入 $O(10^{-4})$ 量级的漂移。这看似无关紧要，但在 GRPO 等算法中会造成实质性的影响。

具体来说，GRPO 的策略梯度依赖 importance ratio $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$。在 On-Policy 场景下，rollout 是由当前策略 $\pi_\theta$ 自身生成的，因此 $\pi_\theta = \pi_{\text{old}}$，理论上 $r(\theta) = 1$，对应 log ratio $\log \pi_\theta - \log \pi_{\text{old}} = 0$。然而，如果推理引擎计算的 $\log \pi_{\text{old}}$ 和训练引擎重新计算的 $\log \pi_\theta$ 之间存在数值差异 $\epsilon$，那么 log ratio 就不再是 0，而是 $\epsilon$。这个 $\epsilon$ 会直接进入梯度估计——GRPO 使用 clipped surrogate objective $\min(r(\theta) \hat{A}, \text{clip}(r(\theta), 1-\epsilon_c, 1+\epsilon_c) \hat{A})$，当 $r(\theta) \neq 1$ 时，clip 机制可能被错误触发或错误绕过，导致策略更新方向偏离真实梯度。更严重的是，这种偏差是系统性的——它不会随着样本增多而被平均掉，而是在每一步训练中都持续引入噪声，最终累积为不可忽略的训练不稳定性。

True On-Policy 的目标很直接：**让推理和训练的 log probability 完全一致，diff = 0，bitwise equal。** 在 NVIDIA GPU 上，这已经通过 Flash Attention 3 + DeepGEMM + batch invariant ops 实现了。但在 AMD GPU 上——没有 FA3，没有 DeepGEMM，一切要从 Triton 开始重建。

**这就是本文解决的问题——我们在 AMD/ROCm 上基于 Triton backend 实现了完整的 True On-Policy 支持，最终达到了 `train_rollout_logprob_abs_diff = 0.0`。**

目前已向 SGLang 和 Miles 提交 PR，正在 review 中。

## 前置知识：NVIDIA 上的 True On-Policy 是怎么做的

在 AMD 上做 True On-Policy 之前，先理解 NVIDIA 版本依赖的基础设施——这些正是我们的出发点，也是我们需要在 AMD 上重建的东西。

NVIDIA 版本的 True On-Policy 建立在三大基石之上。

**Flash Attention 3**。SGLang 在 FA3 backend 上已经实现了确定性推理——推理侧的 decode 和 extend 共享 FA3 的同一套底层实现，天然保证了 attention 计算 bitwise 相同。训练侧同样使用 FA3（通过 `--attn-implementation flash_attention_3`），这样推理和训练的 attention 计算就完全对齐了。这是 True On-Policy 最关键的前提。

**DeepGEMM**。DeepSeek 开源的确定性矩阵乘法库，通过固定 tensor core 指令的调度顺序来消除浮点累积的不确定性，确保 GEMM 计算在不同 batch size 下结果一致。

**Batch Invariant Ops**。Thinking Machines Lab 提出并实现的一组 batch-invariant kernel（`batch_invariant_ops`），通过 monkey patch 的方式替换 LayerNorm、softmax 等算子，使它们的输出不随 batch size 变化。这解决了一个核心问题：推理时多条序列组成一个大 batch 做 forward，而训练时每条序列单独做 forward——如果 LayerNorm 的 reduce 操作因 batch size 不同而导致浮点累积顺序不同，就会引入数值差异。`batch_invariant_ops` 通过将每条序列的计算完全隔离，消除了这个差异来源。

在这三大基石之上，SGLang 又做了一系列精细的工程对齐（确定性 NCCL、RoPE 精度控制等），最终在 NVIDIA GPU 上实现了 `train_rollout_logprob_abs_diff = 0.0`。

同时，SGLang 社区已经在 Triton backend 上实现了确定性推理——`extend_attention_fwd_unified` 这个 1-stage unified kernel 就是在此过程中被引入的，它将 prefix 和 extend 的 KV 合并为统一索引，保证了确定性。**我们的工作正是在 FA3 确定性推理和 Triton 确定性推理这两项已有成果的基础上，进一步将 True On-Policy 扩展到 AMD/ROCm 平台。**

## 为什么 AMD 上没法直接做

理解了 NVIDIA 版本的方案后，就能看清 AMD 上的困难所在——上述三大基石在 AMD/ROCm 上全部失效。

**没有 FA3**。AMD 上没有 Flash Attention 3。虽然 AMD 有自己的高性能算子库 AITER，但 AITER 目前不支持确定性推理，无法保证相同输入产生 bitwise 相同的输出。因此，在 SGLang 的 AMD 支持中，目前唯一能用于确定性推理的 attention backend 就是 Triton。而 SGLang 的 Triton backend 在推理时 decode 和 extend 走的是完全不同的 kernel：decode 走 `decode_attention_fwd()`（paged attention），extend 走 `extend_attention_fwd()`（2-stage kernel）。训练侧（FSDP）默认使用 HuggingFace 的 `flash_attention_2`——这是第三个完全不同的实现。三方用三个不同的 kernel，天然不对齐。

**没有 DeepGEMM**。DeepGEMM 依赖 NVIDIA 的 CUTLASS 和 tensor core，在 AMD 上无法运行，矩阵乘法的确定性需要另寻出路。

**Batch Invariant Ops 覆盖不全**。`batch_invariant_ops` 的 monkey patch 只覆盖了 CUDA 路径（`forward_cuda()`），但 AMD 上 RMSNorm 实际走的是 `forward_aiter()` 或 `forward_hip()`——patch 根本没生效。

**简而言之，NVIDIA 上已经铺好的路，在 AMD 上全部要从零开始用 Triton 重建。**

## 核心方案：统一 Attention Kernel

### 问题分析

先来理清 SGLang Triton backend 中 attention 的原始调用链。推理时，decode 和 extend 走的是**完全不同的两个 kernel**：

- **extend（prefill）阶段**：`forward_extend()` → 调用 `extend_attention_fwd()`，这是一个 2-stage 的 Triton kernel（底层 `_fwd_kernel()`），先分块计算 attention，再合并
- **decode 阶段**：`forward_decode()` → 调用 `decode_attention_fwd()`，这是一个独立的 paged attention decode kernel（底层 `_fwd_kernel_stage1()`），针对单 token decode 优化

而训练侧（FSDP）默认使用 HuggingFace 的 `flash_attention_2` 作为 attention backend——这是第三个完全不同的实现。NVIDIA 版本的 True On-Policy 通过将推理和训练两侧都统一到 FA3 来解决这个问题，但 AMD 上没有 FA3，我们需要找到一个三方都能共用的 kernel。

### 解决方案：统一到 `extend_attention_fwd_unified`

我们选择了 `extend_attention_fwd_unified`——一个 1-stage 的统一 Triton kernel（底层 `_fwd_kernel_unified()`）作为三方共用的唯一 attention 实现。相比原始的 2-stage `extend_attention_fwd`，这个 unified 版本将 prefix 和 extend 的 KV 合并为统一的索引，在一个 stage 内完成计算，天然保证了确定性。

**推理侧 extend**：当 `enable_deterministic=True` 时，`forward_extend()` 路由到 `_forward_extend_unified()`，调用 `extend_attention_fwd_unified`（替代原来的 2-stage `extend_attention_fwd`）。

**推理侧 decode**：decode 在数学上就是退化的 extend——每条序列只有 1 个新 token。我们新增了 `_forward_decode_unified()` 方法，将 decode 请求重新组织成 extend 的格式，同样调用 `extend_attention_fwd_unified`（替代原来的 `decode_attention_fwd`）：

```python
def forward_decode(self, q, k, v, layer_id, ...):
    if self.enable_deterministic:
        return self._forward_decode_unified(q, k, v, layer_id, ...)
    # original decode path: decode_attention_fwd()
    ...
```

**训练侧 FSDP**：我们实现了一个 HuggingFace attention bridge，将 `extend_attention_fwd_unified` 注册为 HF 的自定义 attention backend（替代原来的 `flash_attention_2`）。

### Unified Decode 的性能代价

统一 kernel 带来了对齐，但 decode 阶段会有性能损失。原因在于两个 kernel 的并行策略完全不同：`decode_attention_fwd` 知道 Q 只有 1 个 token，因此将 KV cache 切成多份（Split-KV），由多个 thread block 并行扫描，最后 reduce 合并；而 `extend_attention_fwd_unified` 的并行维度是 Q 的分块数——decode 时 Q 只有 1 个 token，第三维 grid = 1，意味着一个 thread block 串行遍历全部 KV cache。

我们在 H200 上实测了两个 kernel 的 decode 延迟（GQA，16 query heads / 8 kv heads，head_dim=64）：

| seq_len | decode kernel | unified kernel | slowdown |
|---------|--------------|----------------|----------|
| 512     | 0.054 ms     | 0.037 ms       | 0.69x    |
| 1024    | 0.053 ms     | 0.066 ms       | 1.25x    |
| 4096    | 0.052 ms     | 0.248 ms       | 4.78x    |
| 16384   | 0.055 ms     | 1.064 ms       | 19.2x    |

短序列时 unified 反而略快——decode kernel 的 split + reduce 两阶段有固定开销，KV 量少时并行收益不够覆盖。但从 1k token 开始 unified 就明显变慢，且差距随序列长度线性增长：decode kernel 的耗时几乎不随 seq_len 变化（Split-KV 并行消化了增长），而 unified kernel 每翻倍 seq_len 耗时也翻倍。

不过需要注意，decode attention 只是整个 decode step 的一部分（还有 Linear、LayerNorm、MLP 等），且 RL 训练中 decode 的总时间占比取决于序列长度和 batch 大小。实际的端到端性能影响需要在完整训练流程中评估。

这样，**推理的 decode、推理的 extend、训练的 forward 三方统一到同一个 `extend_attention_fwd_unified` kernel，从根源消除差异。**

### 训练侧：SGLang Triton Attention Bridge

具体来说，我们实现了 `apply_sglang_triton_attention_patch`，将 SGLang 的 `extend_attention_fwd_unified` kernel 注册为 HuggingFace 的自定义 attention 实现：

```python
def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton attention as HF custom attention backend."""
    ALL_ATTENTION_FUNCTIONS["sglang_triton"] = _sglang_triton_attention
    model.config._attn_implementation = "sglang_triton"
```

在 `_sglang_triton_attention` 中，我们将 HF 格式的 Q/K/V（`[B, num_heads, S, D]`）reshape 为 SGLang 格式（`[total_tokens, num_heads, D]`），cast 到 bf16，然后调用 `extend_attention_fwd_unified`：

```python
def _sglang_triton_attention(query, key, value, ...):
    # Reshape: [B, num_heads, S, D] -> [total_tokens, num_heads, D]
    q = query.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
    k = key.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_dim)
    v = value.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_dim)

    # Cast to bf16 for kernel compatibility
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    # Use per-request indptrs for batch-invariant computation
    extend_attention_fwd_unified(q, o, k, v, qo_indptr, kv_indptr, ...)
    return attn_output
```

关键点在于使用 **per-request indptrs**——每条序列独立计算，不受 batch composition 影响，从根本上保证了 batch invariance。

## 逐层对齐：七层地狱

统一 attention kernel 只是第一步。推理和训练之间，从 embedding 到 logits 的每一步都可能引入数值差异。我们建立了系统性的逐层 tensor dump 框架，按照 **norm → RoPE → attention → residual → MLP → logits** 的顺序逐一排查并修复。

### 1. LayerNorm：AMD 特有路径的 Patch

AMD/ROCm 上的 RMSNorm 有 `forward_aiter()` 和 `forward_hip()` 两条特有路径，绕过了 CUDA 版 `batch_invariant_ops` 的 patch。修复方式：在这两个函数入口检测 `batch_invariant_mode`，回退到 `forward_native()`。

```python
def forward_cuda(self, x, residual=None, post_residual_addition=None):
    if is_batch_invariant_mode_enabled():
        if (residual is not None
            or get_global_server_args().rl_on_policy_target == "fsdp"):
            return self.forward_native(x, residual, post_residual_addition)
        return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)
```

同时，为了精度一致，我们在 on-policy 模式下：
- 将 RMSNorm 的 weight 保持为 **fp32**
- 将 residual 保持为 **fp32**（`fp32_residual=True`）
- 在乘 weight 之前先将 x cast 回原始 dtype（`cast_x_before_out_mul=True`）

```python
norm_kwargs = dict(
    weight_dtype=torch.float32,
    cast_x_before_out_mul=True,
    override_orig_dtype=torch.float32,
    fp32_residual=True,
) if get_global_server_args().rl_on_policy_target is not None else {}
```

### 2. RoPE：CPU 计算 + 禁用 compile

RoPE 的数值差异来自两个地方：

**问题一**：`inv_freq` 在 GPU 上计算时，浮点精度和 HuggingFace 的 CPU 实现不同。

**修复**：在 on-policy 模式下，在 CPU 上计算 `inv_freq`，再搬到 GPU：

```python
def _compute_inv_freq(self, base):
    init_device = (
        "cpu" if get_global_server_args().rl_on_policy_target is not None
        else None
    )
    inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2,
                                             dtype=torch.float, device=init_device)
                                / self.rotary_dim))
    if get_global_server_args().rl_on_policy_target is not None:
        inv_freq = inv_freq.cuda()
    return inv_freq
```

**问题二**：`torch.compile` 在 AMD 上对 RoPE 引入 nondeterminism。

**修复**：在 on-policy 模式下使用 native forward，同时对 rotary embedding 的应用函数启用 `torch.compile`（这一步是安全的）：

```python
if get_global_server_args().rl_on_policy_target is not None:
    self._forward_method = self.forward_native
    self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
        self._apply_rotary_emb_wrapped
    )
```

### 3. Residual：Deferred Addition 对齐

Layer 1 开始出现 hidden_states 和 residual 的差异。原因是 HuggingFace 使用 deferred residual addition（先算完 attention，最后一次性加上 residual），而 SGLang 某些路径提前做了 residual add。

**修复**：引入 `post_residual_addition` 参数，将 deferred residual 传递到 LayerNorm 中融合计算：

```python
class Qwen3DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch,
                residual, post_residual_addition=None):
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch,
            post_residual_addition=post_residual_addition,
        )
        ...
```

在 `forward_native` 中，residual addition 的顺序被严格控制：

```python
def forward_native(self, x, residual=None, post_residual_addition=None):
    if residual is not None:
        x = x + residual.to(torch.float32)
        if post_residual_addition is not None:
            x = x + post_residual_addition.to(torch.float32)
        if self.fp32_residual:
            residual = x.clone()
    ...
```

### 4. Activation & MLP：统一到 Native 实现

SiLU activation 在不同后端有不同的 fused 实现，可能引入数值差异。修复方式是在 on-policy 模式下统一使用 native 实现：

```python
class SiluAndMul(nn.Module):
    def __init__(self):
        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native
```

MLP 层的输入也需要统一 cast 到 bf16：

```python
class Qwen2MLP(nn.Module):
    def forward(self, x):
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()
        gate_up, _ = self.gate_up_proj(x)
        ...
```

### 5. Logits & Log Probability：bf16 MatMul + fp32 Log-Softmax

**Logits 计算**：由于 tied weight 的存在，lm_head 的 weight dtype 可能不受控。我们在 on-policy 模式下显式 cast 到 bf16 做 matmul：

```python
elif get_global_server_args().rl_on_policy_target is not None:
    logits = torch.matmul(
        hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
    )
```

**Log probability 计算**：bf16 的 `log_softmax` 会引入 ~5e-4 的系统性漂移。修复：在 on-policy 模式下，先将 logits 转为 bf16 做 temperature scaling，再用 `torch.log_softmax` 计算（其内部会提升到 fp32）：

```python
if get_global_server_args().rl_on_policy_target is not None:
    logits_div_temperature = (
        logits.bfloat16().div(sampling_info.temperatures).bfloat16()
    )
    logprobs_via_logsoftmax_kernel = torch.log_softmax(
        logits_div_temperature, dim=-1
    )
```

### 6. Embedding：fp32 参数

Embedding 层的参数在 on-policy 模式下保持 fp32，确保与训练侧一致：

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size, config.hidden_size,
    params_dtype=(
        torch.float32
        if get_global_server_args().rl_on_policy_target is not None
        else None
    ),
)
```

## 调试方法论

整个对齐过程中，我们建立了一套系统性的调试方法：

### 逐层 Tensor Dump

同时运行 SGLang 推理（decode 路径）和 HuggingFace FSDP 训练（extend 路径），在每一层关键节点 dump 中间 tensor：

1. **`q_post_norm` / `k_post_norm`** —— RMSNorm 之后
2. **`q_post_rope` / `k_post_rope`** —— RoPE 之后
3. **`attn_context_before_o_proj`** —— attention 输出、o_proj 之前

使用 SGLang 内置的 `dumper` 工具（`SGLANG_DUMPER_ENABLE=1`）自动 dump，然后离线比较两边的 abs diff。

### Debug 模式

通过环境变量控制粒度：
- `debug_one_sample`：单条序列、2 token 输出，快速定位
- `debug_long`：单条序列、64 token 输出，验证长序列
- `normal`：完整训练验证

### 排查顺序

**Norm → RoPE → Attention → Residual → MLP → Logits → LogProb**

每修复一层，验证该层输出 diff = 0 后再进入下一层。这种逐层递进的方式避免了上游差异掩盖下游 bug。

## 实验结果

我们在 AMD GPU（ROCm）上基于 Qwen3-0.6B 模型进行验证，使用 Miles RL 框架（GRPO on GSM8K）。

### 数值对齐

| 指标 | 结果 |
|------|------|
| `train_rollout_logprob_abs_diff` | **0.0** |
| `mismatch_k3_kl` | **0.0** |

**推理和训练的 log probability 完全一致，bitwise equal。**

### 关键配置

```python
# 推理侧
"--sglang-attention-backend triton"          # Triton attention
"--sglang-enable-deterministic-inference"     # 确定性推理
"--sglang-rl-on-policy-target fsdp"          # 启用 on-policy
"--sglang-disable-cuda-graph"                # 禁用 CUDA graph
"--sglang-disable-radix-cache"               # 禁用 radix cache

# 训练侧
"--attn-implementation triton"               # SGLang Triton bridge
"--deterministic-mode"                       # 确定性模式
"--true-on-policy-mode"                      # On-policy 模式

# 环境变量
NCCL_ALGO=allreduce:tree                     # 确定性 NCCL
SGLANG_RETURN_ORIGINAL_LOGPROB=1             # 返回 pre-softmax logits
```

## 工程实践总结

### 改动文件一览

| 文件 | 核心内容 |
|------|----------|
| `triton_backend.py` | `_forward_decode_unified()` —— 统一 decode/extend kernel 路由 |
| `layernorm.py` | AMD 路径的 `batch_invariant` 检测，fp32 weight/residual |
| `rotary_embedding.py` | 禁用 compile，CPU 上算 cos/sin cache |
| `sampler.py` | fp32 `log_softmax` |
| `activation.py` | SiLU 回退 native 实现 |
| `logits_processor.py` | bf16 matmul 对齐 |
| `qwen2.py` / `qwen3.py` | fp32 norm/residual，bf16 casting，deferred residual |
| `hf_sglang_triton_patch.py` | 训练侧 Triton attention bridge |

### 设计原则

1. **最小侵入**：所有改动都通过 `rl_on_policy_target` flag 控制，不影响正常推理/训练路径
2. **同一 kernel**：推理和训练使用完全相同的 `extend_attention_fwd_unified`，从根源消除差异
3. **Batch Invariant**：per-request indptrs 确保计算与 batch composition 无关
4. **精度控制**：在每一层精确控制 dtype（fp32 用于 accumulation，bf16 用于 compute），与训练侧严格对齐

## 后续工作

当前的实现已经在 AMD 上达成了 True On-Policy 的核心目标（logprob diff = 0），但仍有不少优化空间：

- **AITER 确定性推理支持**：目前 AMD 上只能使用 Triton backend 做确定性推理，而 AITER 作为 AMD 原生的高性能算子库，在非确定性场景下性能显著优于 Triton。如果 AITER 后续支持确定性模式，将 True On-Policy 的 attention kernel 切换到 AITER 可以带来可观的推理加速。
- **确定性推理性能优化**：当前的 unified decode path 将 decode 路由到 extend kernel，牺牲了 decode 阶段的专用优化（如 paged attention 的访存模式）。后续可以探索在保证确定性的前提下，针对 decode 场景优化 unified kernel 的性能。
- **更多模型支持**：目前验证了 Qwen3-0.6B，后续需要扩展到更大规模的模型（Qwen3-4B、Qwen3-32B 等）以及 MoE 架构，验证在不同模型结构下的对齐效果和性能表现。
- **DeepGEMM 替代方案**：AMD 上缺少确定性 GEMM 库，目前依赖 PyTorch 默认的 matmul 实现。后续可以探索基于 Triton 或 composable kernel 实现确定性 GEMM，进一步收紧数值对齐的保证。
- **开启 True On-Policy 的性能开销评估**：需要对比开启和关闭 True On-Policy 模式下的端到端训练吞吐，量化 unified decode path、native fallback、fp32 accumulation 等改动带来的性能开销，为后续优化提供 baseline。

## 致谢

感谢 @Zhiyao Jiang 的协作开发

感谢 @fzyzcjy 在 True On-Policy NVIDIA 版本上的开创性工作
感谢 @yushen在amd sglang 实现的确定性推理
感谢 @jiajun Li @ Yuzhen zhou 的支持

感谢 SGLang 和 Miles 社区的技术支持

2026/3/19
