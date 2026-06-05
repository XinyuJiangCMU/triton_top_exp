#!/usr/bin/env python3
"""
DeepSeek-style block-scale FP8 GEMM in Triton — the kernel ROCm/TransformerEngine is missing.

Gap (rocm_gemm.cu): tensorwise(mode0) + MXFP8(mode1) handled; NVTE_BLOCK_SCALING_1D/2D
(mode 2/3, DeepSeek 1x128 act + 128x128 weight, FP32 scales) falls into
`NVTE_ERROR("unsupported scaling mode")`, and hipBLASLt has no BLK128x128_32F scale mode.
So we compute it ourselves.

C[M,N] = dequant(A_fp8[M,K], sa[M,K/128]) @ dequant(B_fp8[N,K], sb[N/128,K/128])^T  -> bf16

Forward only (miles' FP8 training keeps backward in BF16). Correctness-first
(cast fp8->bf16 for the MMA; native fp8-mma is a perf follow-up).
"""
import torch, triton, triton.language as tl

E4M3_MAX = 448.0
BLK = 128

# ---------- reference: DeepSeek blockwise quantization (torch) ----------
def quant_1x128(A):                       # A[M,K] bf16 -> (fp8[M,K], scale[M,K/128])
    M, K = A.shape
    Ab = A.reshape(M, K // BLK, BLK).float()
    scale = (Ab.abs().amax(dim=2, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    Aq = (Ab / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return Aq.reshape(M, K), scale.squeeze(-1).contiguous()      # scale[M, K/128]

def quant_128x128(B):                     # B[N,K] bf16 -> (fp8[N,K], scale[N/128,K/128])
    N, K = B.shape
    Bb = B.reshape(N // BLK, BLK, K // BLK, BLK).float()
    scale = (Bb.abs().amax(dim=(1, 3), keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    Bq = (Bb / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return Bq.reshape(N, K), scale.reshape(N // BLK, K // BLK).contiguous()

def dequant_A(Aq, sa):
    M, K = Aq.shape
    return (Aq.float().reshape(M, K // BLK, BLK) * sa[:, :, None]).reshape(M, K)

def dequant_B(Bq, sb):
    N, K = Bq.shape
    return (Bq.float().reshape(N // BLK, BLK, K // BLK, BLK) * sb[:, None, :, None]).reshape(N, K)

# ---------- the Triton kernel ----------
@triton.jit
def _bw_fp8_gemm(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr,
                 M, N, K,
                 stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                 stride_sam, stride_sak, stride_sbn, stride_sbk,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M                          # M (tokens) may not be a multiple of 128
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    nb = pid_n                                   # BLOCK_N == 128 -> one weight n-block
    for kb in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)          # [BM,BK] fp8
        b = tl.load(b_ptrs)                      # [BN,BK] fp8
        sa = tl.load(sa_ptr + offs_m * stride_sam + kb * stride_sak, mask=m_mask, other=0.0)
        sb = tl.load(sb_ptr + nb * stride_sbn + kb * stride_sbk)       # scalar (128x128 block)
        # fp8 -> bf16 is lossless (e4m3 subset of bf16); MMA accumulates in f32
        p = tl.dot(a.to(tl.bfloat16), tl.trans(b).to(tl.bfloat16))     # [BM,BN]
        acc += p * sa[:, None] * sb
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])

def bw_fp8_gemm(Aq, sa, Bq, sb):
    M, K = Aq.shape; N, _ = Bq.shape
    C = torch.empty((M, N), dtype=torch.bfloat16, device=Aq.device)
    grid = (triton.cdiv(M, BLK), triton.cdiv(N, BLK))
    _bw_fp8_gemm[grid](
        Aq, Bq, C, sa, sb, M, N, K,
        Aq.stride(0), Aq.stride(1), Bq.stride(0), Bq.stride(1), C.stride(0), C.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        BLOCK_M=BLK, BLOCK_N=BLK, BLOCK_K=BLK)
    return C

# ---------- correctness harness ----------
def main():
    torch.manual_seed(0)
    M, N, K = 512, 512, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    Aq, sa = quant_1x128(A)
    Bq, sb = quant_128x128(B)

    C_tri  = bw_fp8_gemm(Aq, sa, Bq, sb).float()
    C_ref  = dequant_A(Aq, sa) @ dequant_B(Bq, sb).T      # exact fp8-dequant matmul
    C_true = A.float() @ B.float().T                       # pre-quantization ground truth

    def rel(x, y):
        return ((x - y).norm() / y.norm()).item()

    print(f"dims M={M} N={N} K={K}, blocks: A 1x{BLK}, B {BLK}x{BLK}")
    print(f"[kernel correctness]  Triton vs fp8-dequant-matmul : rel_err = {rel(C_tri,  C_ref):.3e}")
    print(f"[quantization error]  Triton vs true bf16 matmul    : rel_err = {rel(C_tri,  C_true):.3e}")
    print(f"max|Triton-ref| = {(C_tri-C_ref).abs().max().item():.3e},  max|C_true| = {C_true.abs().max().item():.1f}")
    ok = rel(C_tri, C_ref) < 1e-2
    print("RESULT:", "PASS (kernel matches fp8 reference)" if ok else "FAIL")

if __name__ == "__main__":
    main()
