"""
================================================================================
Tutorial: 从零开始写 Triton Causal Self-Attention Forward + Backward
================================================================================

这篇教程把 SGLang 的 extend_attention_fwd_unified（forward）和我们自己写的
triton_attn_bwd（backward）从最基础开始讲清楚。

目标：读完这个文件后，你可以自己从头写出这两个算子。

前置知识：
  - 知道 attention 的数学公式（Q @ K^T, softmax, 乘 V）
  - 知道 PyTorch 的 autograd 基本概念
  - 不需要 Triton 经验，会从头解释

================================================================================
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 0: 为什么需要 Triton 版本？                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PyTorch 标准实现:
#     attn = Q @ K^T * scale    # [B, H, S, S]  ← S×S 矩阵，显存 O(S²)
#     attn = softmax(attn)
#     O = attn @ V
#
# 问题：S=4096 时，S×S = 16M 个 float32 = 64MB PER HEAD PER BATCH。
# 一个 32 head 的模型，batch=4：64 * 32 * 4 = 8GB，只是存 attention 矩阵。
#
# 解决：Flash Attention / Triton Tiled Attention
# 核心思想：不存 S×S 矩阵，用 online softmax 逐 block 计算。
# 显存从 O(S²) 降到 O(S)。


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 1: 理解 Triton 基础                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Triton 的编程模型：
#   - 你写一个 "kernel"（用 @triton.jit 修饰的函数）
#   - Kernel 的一次执行叫一个 "program instance"
#   - 通过 tl.program_id(axis) 区分不同 instance
#   - 每个 instance 处理一个 "tile"（数据块）
#
# 类比 CUDA:
#   CUDA:   threadIdx.x, blockIdx.x, blockDim.x
#   Triton: tl.program_id(0), grid[0], BLOCK_SIZE
#
# 关键操作:
#   tl.load(ptr, mask)     - 从 GPU 显存加载到 SRAM（寄存器/shared memory）
#   tl.store(ptr, val)     - 从 SRAM 写回显存
#   tl.dot(a, b)           - 矩阵乘法（在 SRAM 中完成，极快）
#   tl.exp, tl.max, tl.sum - 逐元素或归约操作
#
# 核心模式：Load tile → Compute → Store tile


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 2: Online Softmax — 整个算法的核心 trick                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 问题：softmax(x) = exp(x_i) / sum(exp(x_j))
# 需要先扫一遍所有 x 算 sum，才能归一化。
# 但我们不想存整行的 attention scores（那就是 O(S²) 了）。
#
# 解决：Online Softmax（Milakov & Gimelshein, 2018）
# 维护三个 running 变量：
#   m = 当前见过的最大值
#   l = 累积的 exp 之和（已经 rescale 过的）
#   acc = 累积的加权 V（已经 rescale 过的）
#
# 每来一个新 block:
#   1. 算 block 内的 max: block_max
#   2. 新的全局 max: new_m = max(m, block_max)
#   3. Rescale 旧的 sum:  l = l * exp(m - new_m)
#   4. 加上新 block 的贡献: l += sum(exp(qk - new_m))
#   5. Rescale 旧的 acc:  acc = acc * exp(m - new_m)
#   6. 加上新 block 的贡献: acc += exp(qk - new_m) @ V
#   7. 更新 m = new_m
#
# 最后: O = acc / l
#
# 注意：这给出的结果和标准 softmax 完全一样（数学等价），
# 只是计算顺序不同（streaming / online）。
#
# 下面用 Python 演示：

def online_softmax_attention_demo(Q, K, V, scale):
    """
    用 online softmax 逐 block 计算 attention，不需要 S×S 矩阵。
    Q: [S_q, D], K: [S_kv, D], V: [S_kv, D]
    """
    import torch

    S_q, D = Q.shape
    S_kv = K.shape[0]
    BLOCK_N = 64  # 每次处理 64 个 K/V

    # Running state for each query position
    m = torch.full((S_q,), float("-inf"))  # 当前最大值
    l = torch.zeros(S_q)                   # 累积 exp sum
    acc = torch.zeros(S_q, D)              # 累积加权 V

    for start_n in range(0, S_kv, BLOCK_N):
        end_n = min(start_n + BLOCK_N, S_kv)
        k_block = K[start_n:end_n]  # [block, D]
        v_block = V[start_n:end_n]  # [block, D]

        # Step 1: Q @ K^T for this block
        qk = Q @ k_block.T * scale  # [S_q, block]

        # (这里省略 causal mask，后面会加)

        # Step 2: Online softmax update
        block_max = qk.max(dim=1).values             # [S_q]
        new_m = torch.maximum(m, block_max)           # [S_q]

        # Step 3: Rescale old values
        old_scale = torch.exp(m - new_m)              # [S_q]
        l = l * old_scale                             # rescale old sum
        acc = acc * old_scale.unsqueeze(1)            # rescale old acc

        # Step 4: Add new block contribution
        p = torch.exp(qk - new_m.unsqueeze(1))       # [S_q, block]
        l += p.sum(dim=1)                             # accumulate sum
        acc += p @ v_block                            # accumulate weighted V

        # Step 5: Update running max
        m = new_m

    # Final: normalize
    O = acc / l.unsqueeze(1)
    return O


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 3: Forward Kernel — 逐行解读 _fwd_kernel_unified                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# SGLang 的 _fwd_kernel_unified 本质就是上面 online_softmax_attention_demo
# 的 Triton 版本。去掉 SGLang 特有的功能（sliding window, logit cap, sink
# tokens, custom mask, xai temperature），核心逻辑如下：
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Grid: (batch_size, num_heads, cdiv(S_q, BLOCK_M))                 │
# │                                                                   │
# │ 每个 program instance 处理:                                        │
# │   - 一个 sequence (cur_seq = program_id(0))                        │
# │   - 一个 head    (cur_head = program_id(1))                        │
# │   - BLOCK_M 个 query positions (cur_block_m = program_id(2))       │
# │                                                                   │
# │ 内层循环：遍历所有 K/V blocks                                        │
# └─────────────────────────────────────────────────────────────────────┘
#
# 简化后的核心代码（伪代码对照原始代码）:
#
# ```python
# @triton.jit
# def _fwd_kernel_simplified(Q, O, K, V, qo_indptr, kv_indptr, kv_indices,
#                             prefix_lens, sm_scale, kv_group_num, ...):
#     cur_seq = tl.program_id(0)
#     cur_head = tl.program_id(1)
#     cur_block_m = tl.program_id(2)
#     cur_kv_head = cur_head // kv_group_num  # GQA: 多个 Q head 共享一个 KV head
#
#     # 1. 加载本 sequence 的长度信息
#     q_start = tl.load(qo_indptr + cur_seq)
#     q_len = tl.load(qo_indptr + cur_seq + 1) - q_start
#     kv_start = tl.load(kv_indptr + cur_seq)
#     kv_len = tl.load(kv_indptr + cur_seq + 1) - kv_start
#     prefix_len = tl.load(prefix_lens + cur_seq)
#
#     # 2. 加载 Q tile: [BLOCK_M, D]
#     offs_m = cur_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     mask_m = offs_m < q_len
#     q = tl.load(Q + ..., mask=mask_m)
#
#     # 3. 初始化 online softmax 状态
#     acc = zeros([BLOCK_M, D])     # 累积 P @ V
#     deno = zeros([BLOCK_M])       # 累积 sum(exp)
#     e_max = full([BLOCK_M], -inf) # running max
#
#     # 4. 遍历所有 K/V blocks
#     for start_n in range(0, kv_len, BLOCK_N):
#         # 4a. 通过 kv_indices 查找实际的 K/V 位置
#         kv_loc = tl.load(kv_indices + kv_start + start_n + offs_n)
#
#         # 4b. 加载 K tile: [D, BLOCK_N]（注意转置，方便做 Q @ K^T）
#         k = tl.load(K + kv_loc * stride + cur_kv_head * stride + offs_d)
#
#         # 4c. 计算 attention scores: Q @ K^T * scale
#         qk = tl.dot(q, k) * sm_scale     # [BLOCK_M, BLOCK_N]
#
#         # 4d. Causal mask（仅对 extend 部分，prefix 不需要 mask）
#         k_is_extend = (start_n + offs_n) >= prefix_len
#         k_idx_in_extend = (start_n + offs_n) - prefix_len
#         causal_mask = where(k_is_extend, offs_m >= k_idx_in_extend, True)
#         qk = where(causal_mask, qk, -inf)
#
#         # 4e. Online softmax update（和 Part 2 完全一样）
#         row_max = tl.max(qk, axis=1)
#         n_e_max = maximum(row_max, e_max)
#         re_scale = exp(e_max - n_e_max)
#         p = exp(qk - n_e_max[:, None])
#         deno = deno * re_scale + tl.sum(p, 1)
#
#         # 4f. 加载 V tile 并累积
#         v = tl.load(V + kv_loc * stride + ...)
#         acc = acc * re_scale[:, None] + tl.dot(p, v)
#
#         e_max = n_e_max
#
#     # 5. 归一化并写回
#     tl.store(O + ..., acc / deno[:, None])
# ```
#
# 对照原始代码 (extend_attention.py line 729-940):
#   - line 733-736: program_id, kv_group_num → 对应上面的 1
#   - line 771-777: 加载 Q → 对应上面的 2
#   - line 788-792: 初始化 acc/deno/e_max → 对应上面的 3
#   - line 795: for循环遍历 KV blocks → 对应上面的 4
#   - line 857-861: kv_indices 查找 → 对应 4a
#   - line 864-873: 加载 K → 对应 4b
#   - line 876,890: Q@K^T * scale → 对应 4c
#   - line 816-831: causal mask → 对应 4d
#   - line 900-907: online softmax → 对应 4e
#   - line 909-921: 加载 V + 累积 → 对应 4f
#   - line 931-940: 归一化 + store → 对应 5
#
# SGLang 额外支持的功能（你可以暂时忽略）:
#   - kv_indices: 支持 page table / KV cache（推理用）
#   - prefix_lens: 支持 prefix + extend（推理用）
#   - BLOCK_DPE: 支持 positional encoding 分开的模型（如 DeepSeek）
#   - sliding_window, logit_cap, sink_tokens, xai_temperature: 特定模型需要


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 4: 从零开始写一个最简 Forward Kernel                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 去掉所有花哨功能，只保留 causal attention 的核心：

import torch
import triton
import triton.language as tl


@triton.jit
def _simple_causal_attn_fwd(
    Q, K, V, O,          # 数据指针
    sm_scale,             # = 1/sqrt(D)
    stride_bs, stride_h,  # Q/K/V/O 都是 [total, num_heads, D], stride_bs = num_heads*D
    seqlen, headdim,
    BLOCK_M: tl.constexpr,     # Q tile 大小
    BLOCK_N: tl.constexpr,     # K/V tile 大小
    BLOCK_HEADDIM: tl.constexpr,  # next_power_of_2(D)
):
    """
    最简 causal attention forward。
    Grid: (cdiv(seqlen, BLOCK_M), num_heads)
    每个 instance 处理一个 head 的 BLOCK_M 个 query。
    """
    pid_m = tl.program_id(0)  # 第几个 Q block
    pid_h = tl.program_id(1)  # 第几个 head

    # ─── 加载 Q tile: [BLOCK_M, D] ───
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    q_ptrs = Q + pid_h * stride_h + offs_m[:, None] * stride_bs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # ─── 初始化 online softmax 状态 ───
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)  # 累积 P @ V
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 累积 sum(exp)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running max

    # ─── Causal: Q 位置 i 只能 attend 到 K 位置 0..i ───
    end_n = tl.minimum(seqlen, (pid_m + 1) * BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # ─── 遍历 K/V blocks ───
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seqlen

        # 加载 K tile: [BLOCK_N, D]
        k_ptrs = K + pid_h * stride_h + offs_n_curr[:, None] * stride_bs + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q.to(k.dtype), tl.trans(k)) * sm_scale

        # Causal mask: query i 只能看 key j (j <= i)
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # ─── Online softmax update ───
        block_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, block_max)

        # Rescale 旧的 + 加上新的
        old_scale = tl.exp(m_i - new_m)
        p = tl.exp(qk - new_m[:, None])

        l_i = l_i * old_scale + tl.sum(p, axis=1)

        # 加载 V tile: [BLOCK_N, D]
        v_ptrs = V + pid_h * stride_h + offs_n_curr[:, None] * stride_bs + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        acc = acc * old_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = new_m

    # ─── 归一化 + 写回 ───
    o = acc / l_i[:, None]
    o_ptrs = O + pid_h * stride_h + offs_m[:, None] * stride_bs + offs_d[None, :]
    tl.store(o_ptrs, o.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


def simple_causal_attn_forward(q, k, v):
    """
    Python wrapper.
    q, k, v: [seqlen, num_heads, D], same dtype (bf16)
    returns: o [seqlen, num_heads, D]
    """
    S, H, D = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_HEADDIM = triton.next_power_of_2(D)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), H)
    _simple_causal_attn_fwd[grid](
        q, k, v, o, sm_scale,
        q.stride(0), q.stride(1),
        S, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_HEADDIM=BLOCK_HEADDIM,
        num_warps=4, num_stages=1,
    )
    return o


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 5: Backward — 数学推导                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 已知: Q, K, V, O (forward 的输出), dO (loss 对 O 的梯度)
# 求:   dQ, dK, dV
#
# Forward 的三步:
#   S = Q @ K^T * scale        # attention scores
#   P = softmax(S)             # attention weights
#   O = P @ V                  # output
#
# 反向逐步推导（链式法则，从 O 往前推）:
#
# ━━━ Step 1: dV 和 dP ━━━
#
#   O = P @ V
#   所以: dV = P^T @ dO        (对 V 的梯度)
#         dP = dO @ V^T        (对 P 的梯度)
#
#   直觉: dV_j = "所有 query 传来的梯度，按 attention weight 加权"
#
# ━━━ Step 2: dS（通过 softmax 的 Jacobian）━━━
#
#   P = softmax(S)  →  dS = ?
#
#   Softmax 的 Jacobian:
#     dS_ij = P_ij * (dP_ij - sum_k(P_ik * dP_ik))
#           = P_ij * (dP_ij - delta_i)
#
#   其中 delta_i = sum_k(P_ik * dP_ik)
#               = sum_k(P_ik * dO_i @ v_k^T)
#               = dO_i @ (sum_k P_ik * v_k)^T
#               = dO_i @ O_i^T              ← 因为 O = P @ V !
#               = sum_d(O_id * dO_id)
#
#   所以: delta_i = rowsum(O * dO)  ← 用 forward output 就能算！
#
#   这就是 _bwd_preprocess kernel 算的东西。
#
# ━━━ Step 3: dQ 和 dK ━━━
#
#   S = Q @ K^T * scale
#   所以: dQ = dS @ K * scale     (对 Q 的梯度, 但 scale 已包含在 dS 中)
#         dK = dS^T @ Q * scale   (对 K 的梯度, 同上)
#
#   实际代码中，把 scale 合并到 dS: ds = P * (dP - delta) * scale
#   然后: dQ = ds @ K, dK = ds^T @ Q
#
# ━━━ 汇总: 五个公式 ━━━
#
#   delta_i = sum_d(O_id * dO_id)                    ... (1) _bwd_preprocess
#   dP = dO @ V^T                                    ... (2)
#   dS = P * (dP - delta) * scale                    ... (3)
#   dV = P^T @ dO                                    ... (4)
#   dQ = dS @ K,  dK = dS^T @ Q                      ... (5)
#
# 问题: 公式 (3)(4) 需要 P（attention weights）。
# 但 forward 没有存 P（太大了，S×S）。
# 解决: 重算 P。
#
# ━━━ LSE: 正确重算 P 的关键 ━━━
#
#   P_ij = exp(S_ij) / sum_k exp(S_ik)
#        = exp(S_ij - LSE_i)
#
#   其中 LSE_i = log(sum_k exp(S_ik))   ← log-sum-exp
#
#   有了 LSE，在任何 block 内都可以正确重算 P:
#     已知当前 block 的 S_ij = qk * scale
#     P_ij = exp(S_ij - LSE_i)   ← 全局正确的 attention weight
#
#   如果不用 LSE，而是在每个 block 内独立做 softmax:
#     P_ij = exp(S_ij) / sum_{j in block} exp(S_ij)   ← 分母只有 block 内的
#     这对于多 block 的情况是 ❌ 错误的！
#     （这就是我们第一版的 bug：S > 64 时 grad_norm = inf）


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 6: Backward Kernel 1 — _bwd_preprocess（计算 delta）                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@triton.jit
def _tutorial_bwd_preprocess(
    Out, DO, Delta,
    stride_bs, stride_h,
    seqlen, headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """
    delta_i = sum_d(O_id * dO_id)

    最简单的 kernel：逐位置做点积。
    Grid: (cdiv(seqlen, BLOCK_M), num_heads)
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    # 加载 O 和 dO 的一个 tile: [BLOCK_M, D]
    ptrs = pid_h * stride_h + offs_m[:, None] * stride_bs + offs_d[None, :]
    o = tl.load(Out + ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(DO + ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # 对 D 维度求和: [BLOCK_M]
    delta = tl.sum(o * do, axis=1)

    # 存到 Delta[head, seq]
    tl.store(Delta + pid_h * seqlen + offs_m, delta, mask=mask_m)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 7: Backward Kernel 2 — _compute_lse（计算 LSE）                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# LSE_i = log(sum_{j<=i} exp(q_i @ k_j * scale))
#
# 用 online softmax 的方式计算（和 forward 几乎一样，但不需要 V）：
# 每来一个新 K block:
#   block_max = max(qk)
#   new_m = max(m, block_max)
#   l = l * exp(m - new_m) + sum(exp(qk - new_m))
#   m = new_m
#
# 最后: LSE = m + log(l)

@triton.jit
def _tutorial_compute_lse(
    Q, K, LSE,
    sm_scale,
    stride_bs, stride_h,
    seqlen, headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """
    Grid: (cdiv(seqlen, BLOCK_M), num_heads)
    和 forward 一样的 online softmax，但只算 LSE，不算 O。
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    # 加载 Q: [BLOCK_M, D]
    q_ptrs = Q + pid_h * stride_h + offs_m[:, None] * stride_bs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Online 累积
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    end_n = tl.minimum(seqlen, (pid_m + 1) * BLOCK_M)  # causal bound
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seqlen

        # 加载 K: [BLOCK_N, D]
        k_ptrs = K + pid_h * stride_h + offs_n_curr[:, None] * stride_bs + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Q @ K^T * scale
        qk = tl.dot(q.to(k.dtype), tl.trans(k)) * sm_scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Online update（和 forward 完全一样）
        block_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, block_max)
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(qk - new_m[:, None]), axis=1)
        m_i = new_m

    # LSE = m + log(l)
    lse = m_i + tl.log(tl.where(l_i == 0.0, 1.0, l_i))
    tl.store(LSE + pid_h * seqlen + offs_m, lse, mask=mask_m)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 8: Backward Kernel 3 — _bwd_kernel_dk_dv                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 这个 kernel 计算 dK 和 dV。
#
# 设计: 每个 program instance 拥有一个 K/V block，遍历所有 Q blocks。
# 这样 dK 和 dV 在寄存器中累加，不需要 atomic_add。
#
# Grid: (cdiv(seqlen, BLOCK_N), num_heads)
#
# 伪代码:
#   k, v = load K/V block                # [BLOCK_N, D], 驻留 SRAM
#   dk = zeros, dv = zeros                # 累加器
#
#   for each Q block (causal bound 之后):
#     q = load Q block                    # [BLOCK_M, D]
#     do = load dO block                  # [BLOCK_M, D]
#     lse_i = load LSE                    # [BLOCK_M]
#     delta_i = load Delta                # [BLOCK_M]
#
#     qk = q @ k^T * scale               # [BLOCK_M, BLOCK_N]
#     qk = causal_mask(qk)
#     p = exp(qk - lse_i)                # ← 用 LSE 重算全局正确的 P
#
#     dv += p^T @ do                      # 公式 (4)
#     dp = do @ v^T                       # 公式 (2)
#     ds = p * (dp - delta_i) * scale     # 公式 (3)
#     dk += ds^T @ q                      # 公式 (5)
#
#   store dk, dv
#
# 注意两个关键点:
# 1. p = exp(qk - LSE_i) 而不是 block-local softmax
# 2. 累加器用 float32（即使输入是 bf16）防止精度损失


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 9: Backward Kernel 4 — _bwd_kernel_dq                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 和 _bwd_kernel_dk_dv 的思路对称:
# 每个 program instance 拥有一个 Q block，遍历所有 K/V blocks。
# dQ 在寄存器中累加。
#
# Grid: (cdiv(seqlen, BLOCK_M), num_heads)
#
# 伪代码:
#   q, do = load Q/dO block              # [BLOCK_M, D], 驻留 SRAM
#   lse_i = load LSE, delta_i = load Delta
#   dq = zeros                            # 累加器
#
#   for each K/V block (causal bound 之前):
#     k = load K block, v = load V block  # [BLOCK_N, D]
#
#     qk = q @ k^T * scale               # [BLOCK_M, BLOCK_N]
#     qk = causal_mask(qk)
#     p = exp(qk - lse_i)                # 全局正确的 P
#
#     dp = do @ v^T                       # [BLOCK_M, BLOCK_N]
#     ds = p * (dp - delta_i) * scale     # [BLOCK_M, BLOCK_N]
#     dq += ds @ k                        # [BLOCK_M, D]
#
#   store dq
#
# 为什么需要两个 kernel（dk_dv 和 dq）？
#   - dk_dv: 外层循环是 K block，需要遍历 Q 来累加 dK
#   - dq:    外层循环是 Q block，需要遍历 K 来累加 dQ
#   - 如果用一个 kernel，要么 dK 要么 dQ 需要 atomic_add（慢且不确定性）
#   - 两个 kernel 各自在寄存器累加，无 atomic_add = 更快 + 确定性


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 10: GQA（Grouped Query Attention）处理                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 很多模型的 num_heads != num_kv_heads（例如 Qwen3: 16 heads, 8 kv heads）
# 每 2 个 Q head 共享 1 个 KV head。
#
# Forward: 简单，用 cur_kv_head = cur_head // kv_group_num 找到对应的 KV head。
#
# Backward: 需要注意：
#   - 先把 K/V expand 到 num_heads 维度（让每个 Q head 有自己的 K/V copy）
#   - Triton kernel 按 num_heads 计算 dk/dv
#   - 最后把 dk/dv reshape 回 (num_kv_heads, group_size, D) 然后 sum(dim=group)
#
# 代码中的实现:
#   k_exp = k.view(B, S, nkv, D).unsqueeze(3).expand(B, S, nkv, group, D)
#   k_exp = k_exp.reshape(B*S, num_heads, D).contiguous()
#   ... run triton backward on k_exp ...
#   dk = dk_exp.view(B*S, nkv, group, D).sum(dim=2)  # reduce back


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 11: 完整的正确性验证                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# 验证策略: 用 PyTorch 写一个"笨但正确"的 reference，对比 Triton 结果。
#
# Reference:
#   attn = Q @ K^T * scale          # [B,H,S,S] ← 完整 S×S 矩阵
#   attn.masked_fill_(causal, -inf)
#   attn = softmax(attn)
#   O = attn @ V
#   O.backward(grad_output)
#   → q.grad, k.grad, v.grad       # PyTorch autograd 自动算的
#
# Triton:
#   O = triton_forward(Q, K, V)
#   dQ, dK, dV = triton_backward(Q, K, V, O, grad_output)
#
# 对比指标:
#   - cosine_similarity(triton, reference) > 0.999  ← 方向几乎完全一致
#   - 无 inf / nan
#   - mean_abs_diff 在合理范围（bf16 精度）


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Part 12: 验证代码 — 你可以直接跑                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def verify_tutorial_forward():
    """验证 Part 4 写的简化 forward 和 PyTorch reference 一致。"""
    torch.manual_seed(42)
    S, H, D = 256, 4, 64
    device = "cuda"

    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)

    # Triton forward
    o_tri = simple_causal_attn_forward(q, k, v)

    # PyTorch reference
    q_4d = q.transpose(0, 1).unsqueeze(0).float()  # [1, H, S, D]
    k_4d = k.transpose(0, 1).unsqueeze(0).float()
    v_4d = v.transpose(0, 1).unsqueeze(0).float()
    scale = 1.0 / (D ** 0.5)
    attn = (q_4d @ k_4d.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    o_ref = (attn @ v_4d).squeeze(0).transpose(0, 1)  # [S, H, D]

    cos = torch.nn.functional.cosine_similarity(
        o_tri.float().reshape(-1), o_ref.reshape(-1), dim=0
    )
    print(f"Forward verification: cosine_sim = {cos:.6f} (should be > 0.999)")
    assert cos > 0.999, f"Forward verification failed: cos = {cos}"
    print("✓ Forward PASSED\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Tutorial: Triton Causal Attention Forward + Backward")
    print("=" * 70)
    print()
    verify_tutorial_forward()
    print("读完这个文件，你已经理解了:")
    print("  1. Online softmax 的原理和实现")
    print("  2. Forward kernel 的完整逻辑")
    print("  3. Backward 的数学推导（5 个公式）")
    print("  4. 为什么需要 LSE + 第一版的 bug 在哪")
    print("  5. 两个 backward kernel 的设计（dk_dv vs dq）")
    print("  6. GQA 的处理方式")
    print()
    print("完整的 backward 实现在:")
    print("  miles/backends/fsdp_utils/sglang_attn_bridge/triton_attn_bwd.py")
    print()
    print("完整的 correctness test 在:")
    print("  attention_test/test_triton_attn_bwd.py (20 test configs, all pass)")
