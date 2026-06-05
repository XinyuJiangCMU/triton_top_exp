#!/usr/bin/env python3
"""
Variable-length GROUPED block-scale FP8 GEMM in Triton (the MoE expert kernel).

Forward per expert e:  out[rows_e, :] = act[rows_e] @ weight[e]^T
  act    : [TotM, K]   fp8, 1x128 row scales      sa[TotM, K/128]
  weight : [E, N, K]   fp8, 128x128 block scales   sw[E, N/128, K/128]
Each expert owns a contiguous, 128-padded token span (m_splits). One kernel launch
covers all experts via a flattened (expert, m-tile, n-tile) tile schedule built on host.
"""
import torch, triton, triton.language as tl
from bw_fp8_gemm import E4M3_MAX, BLK, quant_1x128, quant_128x128, dequant_A, dequant_B

@triton.jit
def _grouped_kernel(a_ptr, asc_ptr, w_ptr, wsc_ptr, c_ptr,
                    te_ptr, tm_ptr, tn_ptr, tend_ptr, K,
                    s_am, s_ak, s_asm, s_ask,
                    s_we, s_wn, s_wk, s_wse, s_wsn, s_wsk,
                    s_cm, s_cn, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    e   = tl.load(te_ptr + pid)
    m0  = tl.load(tm_ptr + pid)
    n0  = tl.load(tn_ptr + pid)
    end = tl.load(tend_ptr + pid)                # global row end of this expert's tokens
    offs_m = m0 + tl.arange(0, BLOCK)
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    m_mask = offs_m < end                        # mask the partial last m-tile of an expert
    nb = n0 // BLOCK
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for kb in range(tl.cdiv(K, BLOCK)):
        kk = kb * BLOCK + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * s_am + kk[None, :] * s_ak, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptr + e * s_we + offs_n[:, None] * s_wn + kk[None, :] * s_wk)
        sa = tl.load(asc_ptr + offs_m * s_asm + kb * s_ask, mask=m_mask, other=0.0)   # 1x128
        sw = tl.load(wsc_ptr + e * s_wse + nb * s_wsn + kb * s_wsk)                   # 128x128 scalar
        p = tl.dot(a.to(tl.bfloat16), tl.trans(w).to(tl.bfloat16))
        acc += p * sa[:, None] * sw
    c = c_ptr + offs_m[:, None] * s_cm + offs_n[None, :] * s_cn
    tl.store(c, acc.to(tl.bfloat16), mask=m_mask[:, None])

def grouped_bw_fp8_gemm(a_q, a_sc, w_q, w_sc, m_splits, N):
    """a_q[TotM,K] fp8 + a_sc[TotM,K/128]; w_q[E,N,K] fp8 + w_sc[E,N/128,K/128].
    m_splits are real per-expert token counts (need NOT be multiples of 128)."""
    TotM, K = a_q.shape
    C = torch.empty((TotM, N), dtype=torch.bfloat16, device=a_q.device)
    te, tm, tn, tend = [], [], [], []
    off = 0
    for e, m in enumerate(m_splits):
        nmt = (m + BLK - 1) // BLK                # ceil -> partial last tile masked
        for mt in range(nmt):
            for nt in range(N // BLK):
                te.append(e); tm.append(off + mt * BLK); tn.append(nt * BLK); tend.append(off + m)
        off += m
    if not te:
        return C
    dev = a_q.device
    ti = lambda L: torch.tensor(L, device=dev, dtype=torch.int32)
    _grouped_kernel[(len(te),)](
        a_q, a_sc, w_q, w_sc, C, ti(te), ti(tm), ti(tn), ti(tend), K,
        a_q.stride(0), a_q.stride(1), a_sc.stride(0), a_sc.stride(1),
        w_q.stride(0), w_q.stride(1), w_q.stride(2),
        w_sc.stride(0), w_sc.stride(1), w_sc.stride(2),
        C.stride(0), C.stride(1), BLOCK=BLK)
    return C

# ---------- standalone correctness ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    E, K, N = 4, 512, 512
    m_splits = [100, 256, 130, 384]              # variable per-expert, NOT multiples of 128
    TotM = sum(m_splits)
    A = torch.randn(TotM, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * (K ** -0.5)

    a_q, a_sc = quant_1x128(A)
    w_q = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    w_sc = torch.empty(E, N // BLK, K // BLK, device="cuda")
    for e in range(E):
        q, s = quant_128x128(W[e]); w_q[e], w_sc[e] = q, s

    C = grouped_bw_fp8_gemm(a_q, a_sc, w_q, w_sc, m_splits, N).float()

    # reference: per-expert dequant matmul
    ref = torch.empty(TotM, N, device="cuda")
    off = 0
    for e, m in enumerate(m_splits):
        ad = dequant_A(a_q[off:off+m], a_sc[off:off+m])
        wd = dequant_B(w_q[e], w_sc[e])
        ref[off:off+m] = ad @ wd.T
        off += m
    true = torch.empty(TotM, N, device="cuda")
    off = 0
    for e, m in enumerate(m_splits):
        true[off:off+m] = A[off:off+m].float() @ W[e].float().T; off += m

    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    print(f"experts={E} m_splits={m_splits} K={K} N={N}")
    print(f"[grouped kernel] Triton vs fp8-dequant ref : rel_err = {rel(C, ref):.3e}")
    print(f"[quant error]    Triton vs true bf16       : rel_err = {rel(C, true):.3e}")
    print("RESULT:", "PASS" if rel(C, ref) < 1e-2 else "FAIL")
