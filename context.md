# Triton 对齐交接手册

## 1. 总目标

这条线的最终目标只有一个：

**让 Triton backend 下的 SGLang rollout logprob，与 FSDP/HF 路径重算出来的 logprob 完全一致。**

理想状态分三层：

1. 中间张量一致
2. attention 输出一致
3. 最终 logprob 一致

当前还没有走到第 3 层，甚至第 2 层也还没完全确认。现在主要在做的是：

**把 mismatch 从最终 logprob，一步步缩小到 attention 链路里的具体阶段。**

---

## 2. 当前总体判断

目前已经确认：

- SGLang deterministic triton 路径的 attention 核心目标是 `extend_attention_fwd_unified`
- HF 自定义 triton 路径也在用 `extend_attention_fwd_unified`

所以大方向上，**两边想对齐的是同一个 attention kernel 家族**。

但现在还**不能**下结论说问题就在 attention kernel 本身，因为：

- 最终 logprob 还没对齐
- `attn_context_before_o_proj` 差异很大
- 但 HF 侧关键中间点一度没成功 dump 出来

也就是说，当前阶段的核心问题不是“最终值为什么错”，而是：

**分叉到底发生在 norm、rope，还是 attention kernel 调用阶段。**

---

## 3. 现在这条排查线在做什么

为了定位分叉点，当前把 attention 链路拆成三层来看：

1. `q_post_norm / k_post_norm`
2. `q_post_rope / k_post_rope`
3. `attn_context_before_o_proj`

判断规则：

- 如果 `q_post_norm / k_post_norm` 就不一致
  说明问题在 q/k norm 或更早，attention kernel 暂时不背锅
- 如果 norm 一致，但 `q_post_rope / k_post_rope` 不一致
  说明问题在 rotary embedding 或 position 语义
- 如果 norm 和 rope 都一致，但 `attn_context_before_o_proj` 不一致
  才开始重点怀疑 attention kernel、KV 索引、prefix 语义、causal 语义

---

## 4. 当前已经做过的关键改动

### 4.1 `day0_deterministic_align.py`

文件：

- `/app/true_on_policy/experiment/set_deterministic_args/day0_deterministic_align.py`

已经做过的事情：

- 增加了 HF 最后一层 attention hook 的中间张量 dump
- 增加了分阶段错误提示，不再静默吞错
- 增加了 `--hf-hook-debug`
- 在请求 SGLang `/generate` 前后打印 dump 目录变化：
  - `created_dirs`
  - `touched_dirs`

作用：

- 锁定这次请求实际使用的 `SG_DIR`
- 明确 HF hook 失败到底发生在哪一段

### 4.2 `compare_hf_sglang_step3_attn.py`

文件：

- `/app/true_on_policy/experiment/step1_compare/compare_hf_sglang_step3_attn.py`

已经做过的事情：

- 支持显式传：
  - `--hf-dir`
  - `--sg-dir`
- 支持：
  - `--list-latest`

作用：

- 不再依赖“自动猜最近两个 dump 目录”
- 可以明确指定本轮实验的 HF / SG dump

---

## 5. 目前最新已确认的事实

### 5.1 本轮 SGLang dump 目录怎么判断

现在 `day0_deterministic_align.py` 会打印：

```text
[SGLang dump] created_dirs=[...]
[SGLang dump] touched_dirs=[...]
```

这就是本轮请求真实的 SGLang dump 依据。

### 5.2 最近一次有效运行里确认过的目录

已观察到：

- `SG_DIR=/tmp/dumper/sglang_dump_1772470991.544385`
- `HF_DIR=/tmp/dumper/sglang_dump_1772471004.412183`

注意：

- `SG_DIR` 和 `HF_DIR` 不一定都是“最新两个目录”
- 必须以本轮 `created_dirs / touched_dirs` 和 HF 新输出目录为准

### 5.3 最近一次关键失败点

HF hook 已经给出明确错误：

```text
[HF dump] q_post_norm/k_post_norm failed:
RuntimeError: missing metadata: num_heads=None, num_kv_heads=None, head_dim=128
```

这说明当时的问题不是 rope，不是 attention kernel，而是：

**HF hook 里推断 head 数的方法错了。**

后续已经把逻辑改成：

- `num_heads = q.shape[-1] // head_dim`
- `num_kv_heads = k.shape[-1] // head_dim`

而不是依赖：

- `_module.num_heads`
- `_module.num_key_value_heads`

---

## 6. 当前建议工作流

每次都按下面顺序做，不要跳。

### 步骤 1：确认 SGLang 服务在线

如果 `localhost:30000` 没有服务，后面所有结论都无效。

### 步骤 2：跑最小样本对齐脚本

命令：

```bash
PYTHONPATH=/app/sglang/python:/data/true_on_policy/miles:$PYTHONPATH \
SGLANG_DUMPER_ENABLE=1 \
SGLANG_DUMPER_DIR=/tmp/dumper \
SGLANG_DUMPER_WRITE_FILE=1 \
python /data/true_on_policy/experiment/set_deterministic_args/day0_deterministic_align.py \
  --host localhost \
  --port 30000 \
  --model-path Qwen/Qwen3-8B \
  --attn-implementation triton \
  --dtype bf16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --max-new-tokens 1 \
  --strict-bitwise \
  --hf-hook-debug
```

这一步重点不是看最终 pass/fail，而是看两类输出：

1. `SGLang dump`
2. `HF dump`

### 步骤 3：记录本轮两个目录

记下：

- 本轮 `SG_DIR`
- 本轮 `HF_DIR`

不要靠“最近两个目录”猜。

### 步骤 4：只在 HF 中间点真正产出后，再做 compare

等 HF 侧真实产出这些点后，再 compare：

- `q_post_norm`
- `k_post_norm`
- `q_post_rope`
- `k_post_rope`
- `attn_context_before_o_proj`

命令：

```bash
python /app/true_on_policy/experiment/step1_compare/compare_hf_sglang_step3_attn.py \
  --hf-dir /tmp/dumper/<hf_dir> \
  --sg-dir /tmp/dumper/<sg_dir>
```

---

## 7. 当前最关键的结论

到目前为止，最重要的不是某个数值，而是下面这个定位：

**我们还没有资格判断问题是不是 attention kernel。**

原因：

- 最终 logprob 还不一致
- `attn_context_before_o_proj` 差异很大
- 但 HF 侧 norm / rope 中间点还没有稳定全部产出

所以当前排查重点必须继续放在：

1. 先让 HF 稳定 dump 出 `q_post_norm / k_post_norm`
2. 再让 HF 稳定 dump 出 `q_post_rope / k_post_rope`
3. 然后才有资格判断 `attn_context_before_o_proj` 的差异是不是 kernel 问题

---

## 8. 下一步优先级

按优先级排序：

1. 确认 HF 侧 `q_post_norm / k_post_norm` 已成功产出
2. 确认 HF 侧 `q_post_rope / k_post_rope` 已成功产出
3. 用本轮显式 `HF_DIR / SG_DIR` 做 compare
4. 再根据 compare 结果判断分叉位置：
   - norm
   - rope
   - attn_context

如果这 4 个 HF 点还没产出，就不要急着解释最终 logprob。

