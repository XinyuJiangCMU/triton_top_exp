# HF 与 SGLang 的 RoPE 对齐记录

## 背景

这份文档记录的是一次真实排查过程，目标不是解释 RoPE 理论，而是回答一个工程问题：

**怎样让 HF 对照路径与 SGLang server 路径在 RoPE 这一层尽可能完全对齐，用于 true-on-policy / deterministic alignment。**

这里最关心的不是抽象公式，而是：

- SGLang server 真实运行得到的 `q_post_rope / k_post_rope`
- HF 对照路径重算得到的 `q_post_rope / k_post_rope`

是否一致。

## 最终结论

### 1. 最关键代码改动

必须修改：

- [qwen3.py](/app/sglang/python/sglang/srt/models/qwen3.py)

在 `get_rope(...)` 这一段显式传：

```python
self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=max_position_embeddings,
    base=rope_theta,
    rope_scaling=rope_scaling,
    dtype=torch.float32,
)
```

### 2. 这行为什么重要

这行不是把整个模型改成 `fp32`，它控制的是：

- `rotary_emb`
- RoPE cache
- cos/sin cache

的 dtype。

不加这行时，RoPE cache 会跟默认 dtype 走。在这次 server 场景里，这很容易落到 `bf16`，然后导致：

- server 路径的 rope 输出
- 与 HF / local compare 路径的 rope 输出

出现明显差异。

加上 `dtype=torch.float32` 之后，server 侧 RoPE cache 的初始化语义会和本地复现/HF 对照路径更接近，RoPE 这一层才有机会真正对齐。

### 3. 加了之后观察到的现象

加上这一行后，之前观测到的现象是：

- server dump 的 `cos/sin` 才能和本地 compare 对齐
- rope 从明显偏差，变成可对齐 / 基本对齐

也就是说：

**这是 RoPE 对齐成立的关键前提之一。**

### 4. 去掉会怎样

去掉 `dtype=torch.float32` 后，很可能回到：

- server 的 RoPE cache 使用 `bf16`
- `sg_dump_cos_vs_sg_local_cos/sin != 0`
- rope compare 重新不对齐
- 差异重新从很小尾差，变回明显偏差

所以这行不是“优化项”，而是这次 RoPE 对齐的核心修复。

## 工程目标

我们最终更关心的是：

**HF 与 SGLang server dump 是否对齐。**

不是要求所有 compare 分支都两两完全一致。

尤其在实际验证里，可能出现：

- `HF_MATCH_SGLANG_DUMP: YES`
- 但 `SG_RERUN_MATCH_DUMP: NO`

这种情况并不自动说明失败。因为：

- `server dump` 代表真实 runtime path
- `local SG rerun` 只是脚本里的复现路径

对 true-on-policy 工程目标来说，**HF 对上 server dump 更重要。**

## 关键文件

- [qwen3.py](/app/sglang/python/sglang/srt/models/qwen3.py)
- [day0_deterministic_align.py](/app/true_on_policy/experiment/set_deterministic_args/day0_deterministic_align.py)
- [verify_rope_against_sglang_dump.py](/app/true_on_policy/experiment/step1_compare/verify_rope_against_sglang_dump.py)
- [compare_hf_sglang_step3_attn.py](/app/true_on_policy/experiment/step1_compare/compare_hf_sglang_step3_attn.py)

## 启动命令

### 1. 启动 SGLang server

```bash
CUDA_VISIBLE_DEVICES=7 \
HIP_VISIBLE_DEVICES=7 \
SGLANG_DUMPER_ENABLE=1 \
SGLANG_DUMPER_DIR=/tmp/dumper \
SGLANG_DUMPER_WRITE_FILE=1 \
SGLANG_RETURN_ORIGINAL_LOGPROB=1 \
USE_ROCM_AITER_ROPE_BACKEND=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --attention-backend triton \
  --mem-fraction-static 0.7 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-deterministic-inference \
  --disable-radix-cache \
  --rl-on-policy-target fsdp \
  --skip-server-warmup \
  --disable-cuda-graph
```

和对齐最相关的参数：

- `--rl-on-policy-target fsdp`
  让 SGLang 进入与 FSDP / true-on-policy 对齐的目标路径。
- `--enable-deterministic-inference`
  尽量消掉推理侧非确定性。
- `--disable-radix-cache`
  避免 cache 路径影响对齐判断。
- `--disable-cuda-graph`
  避免 graph 执行路径引入额外变量。
- `USE_ROCM_AITER_ROPE_BACKEND=0`
  禁掉 AITER rope backend，减少 RoPE 后端差异。
- `SGLANG_DUMPER_*`
  用于保存 server 真实运行时的 dump。

### 2. 运行 day0 compare

推荐单卡运行 HF 对照路径，避免 `device_map=auto` 多卡切分带来的 ROCm 不稳定：

```bash
CUDA_VISIBLE_DEVICES=6 \
HIP_VISIBLE_DEVICES=6 \
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
  --use-batch-invariant
```

这个命令的作用：

- 请求 SGLang server 做 rollout
- 在 HF 本地重算同一条链路
- 生成一份 HF dump
- 最后输出 day0 compare 结果

它会产出：

- HF dump 目录，例如 `/tmp/dumper/sglang_dump_1772495870.8373513`
- 结果文件，例如：
  - `results/day0_manifest_*.json`
  - `results/day0_compare_*.json`
  - `results/day0_compare_*_detail.txt`

怎么知道 dump 目录是哪一个：

- 看终端里的 `[Dumper] Choose partial_name=...`
- 或执行：

```bash
ls /tmp/dumper | tail
```

### 3. 运行 rope verify

下面这个命令是一次真实成功复现的例子：

```bash
python /app/true_on_policy/experiment/step1_compare/verify_rope_against_sglang_dump.py \
  --dump-dir /tmp/dumper/sglang_dump_1772495870.8373513 \
  --q-in-name q_post_norm \
  --k-in-name k_post_norm \
  --q-out-name q_post_rope \
  --k-out-name k_post_rope \
  --positions-name layer0_positions \
  --q-in-index 12 \
  --k-in-index 13 \
  --positions-index 3 \
  --q-out-index 14 \
  --k-out-index 15 \
  --verbose
```

各字段的含义：

- `q_post_norm / k_post_norm`
  表示 RoPE 前的 q/k。
- `q_post_rope / k_post_rope`
  表示 RoPE 后的 q/k。
- `layer0_positions`
  这次 rope 使用的位置编码索引。
- `name + index` 同时传
  是为了避免同名 dump 太多时选错文件。
- 这里只看 RoPE 前后
  是因为这里的目标是验证 RoPE 本身，而不是整个 attention 链路。

## 如何判断成功

这次对齐里，最重要的成功标准不是 `SG_RERUN_MATCH_DUMP`，而是：

- `sg_dump_vs_hf_q` `bitwise_equal: True`
- `sg_dump_vs_hf_k` `bitwise_equal: True`
- `HF_MATCH_SGLANG_DUMP: YES`

如果看到这些，说明：

**HF rope 输出已经对上了这次 dump 的 rope 输出。**

### 一个需要特别说明的现象

有时会出现：

- `HF_MATCH_SGLANG_DUMP: YES`
- 但 `SG_RERUN_MATCH_DUMP: NO`

这不代表失败。

它表示：

- server dump 代表真实 runtime path
- verify 脚本里的 local SG rerun 是本地复现路径
- 这两条路径未必完全相同

但从工程目标看，只要：

**HF 对上了 server dump**

就已经比 “HF 对上 local rerun” 更重要。

## 这次真实实验中观察到的关键现象

在 `batch_invariant + bf16` 的某次成功复现里，看到：

- `sg_dump_vs_hf_q` -> `max_abs: 0.0`
- `sg_dump_vs_hf_k` -> `max_abs: 0.0`
- `HF_MATCH_SGLANG_DUMP: YES`

同时：

- `sg_dump_vs_sg_local_q/k = 0.03125`
- `SG_RERUN_MATCH_DUMP: NO`

正确理解应该是：

- 这份 dump 对 HF 路径是完全自洽的
- local SG rerun 没命中，不妨碍 “HF 已和 dump 对齐” 这个更重要的结论成立

## 常见坑

### 1. 不给 `get_rope` 显式传 `dtype=torch.float32`

后果：

- server RoPE cache 很可能回到 `bf16`
- rope compare 重新失去对齐

### 2. 把 HF dump 当成 SG atomic rope dump

不是所有 dump 都能当成 server 原子 rope ground truth。

如果你用的是：

- `layer0_positions`
- `q_post_norm`
- `q_post_rope`

这更像 HF 路径里的中间点，不一定等价于 server runtime rope atomic 调用上下文。

### 3. 多卡 auto-shard 导致 batch-invariant 崩溃

在 ROCm 环境里，`--use-batch-invariant` + 多卡 `device_map=auto` 可能导致崩溃。  
推荐像这次一样：

- server 单独占一张卡
- HF day0 单独占一张卡

### 4. `name` 不传全 / `index` 不传

同名 dump 很多时容易选错文件。  
做精确复现时，建议同时传：

- `--q-in-name`
- `--q-in-index`

这样的组合。

## FAQ / Notes

### Q1. 为什么加 `dtype=torch.float32` 能帮助对齐？

因为它控制的是：

- rotary_emb
- rope cache
- cos/sin cache

的 dtype。

它不是把整个模型改成 `fp32`，只是把 RoPE cache 固定到 `fp32`。  
不加的话，server 侧很容易跟默认 dtype 走到 `bf16`，从而改变 rope 输出。

### Q2. 去掉这行会怎样？

很可能又回到：

- `bf16` rope cache
- server rope 输出重新和 HF 不对齐

这件事在这次排查里已经真实观察到过。

### Q3. 我们最终是不是只关心 HF 和 SGLang server 对齐？

是。

最终目标是 true-on-policy 对齐，不是证明所有 compare 分支都彼此完全一致。

### Q4. 为什么 verify 里 local SG rerun 可能不一致，但 HF 却能和 dump 对齐？

因为：

- local rerun 走的是脚本复现路径
- dump 代表的是 server 真实 runtime path

如果 HF 能直接对上 server dump，那从工程目标看已经更重要。

## 最小复现 Checklist

- [ ] [qwen3.py](/app/sglang/python/sglang/srt/models/qwen3.py) 中 `get_rope` 显式传 `dtype=torch.float32`
- [ ] server 用 deterministic + fsdp target 启动
- [ ] `day0_deterministic_align.py` 成功跑出 dump
- [ ] `verify_rope_against_sglang_dump.py` 指向正确 dump-dir
- [ ] 重点关注 `q_post_norm -> q_post_rope`、`k_post_norm -> k_post_rope`
- [ ] 看到 `HF_MATCH_SGLANG_DUMP: YES`

