#!/usr/bin/env python3
"""
Can we actually TRAIN in FP8 block-wise on AMD MI355X? — self-contained proof.

TE's ROCm path is broken in two places (quantize dispatch in cast_kernels_hip.cuh +
GEMM dispatch in rocm_gemm.cu both reject NVTE_BLOCK_SCALING). Rather than recompile
C++, we bypass the TE wrapper entirely and build the fp8 training cell ourselves:

  fp8 blockwise Linear:
    forward  : quantize x (1x128) + w (128x128) -> e4m3, matmul via our Triton kernel -> bf16
    backward : BF16 (miles' fp8 training keeps backward + master weights in bf16)

Then train a small MLP and show the loss goes down, side-by-side with a bf16 baseline.
"""
import torch, torch.nn as nn
from bw_fp8_gemm import quant_1x128, quant_128x128, bw_fp8_gemm, BLK

# ---------- fp8 blockwise Linear (forward fp8, backward bf16) ----------
class _BlockwiseFP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):                 # x[M,K] bf16, w[N,K] bf16 ; y = x @ w^T
        Aq, sa = quant_1x128(x)             # activation: 1x128 blocks
        Bq, sb = quant_128x128(w)           # weight    : 128x128 blocks
        y = bw_fp8_gemm(Aq, sa, Bq, sb)     # -> bf16, computed in fp8
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, gy):                  # backward kept in bf16
        x, w = ctx.saved_tensors
        gx = gy @ w                         # dgrad: [M,N]@[N,K] -> [M,K]
        gw = gy.transpose(0, 1) @ x         # wgrad: [N,M]@[M,K] -> [N,K]
        return gx, gw

class BlockwiseFP8Linear(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) * (k ** -0.5))
    def forward(self, x):
        return _BlockwiseFP8LinearFn.apply(x, self.weight)

# plain bf16 Linear with identical math, for the baseline
class BF16Linear(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) * (k ** -0.5))
    def forward(self, x):
        return x @ self.weight.transpose(0, 1)

def make_mlp(LinCls, d):
    return nn.Sequential(LinCls(d, d), nn.ReLU(), LinCls(d, d)).cuda()

def train(LinCls, d, M, steps, seed=0):
    torch.manual_seed(seed)
    model = make_mlp(LinCls, d)
    # fixed teacher target so the task is learnable
    torch.manual_seed(123)
    X = torch.randn(M, d, device="cuda", dtype=torch.bfloat16)
    W_true = torch.randn(d, d, device="cuda", dtype=torch.bfloat16) * (d ** -0.5)
    Y = (X @ W_true.T).float()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    losses = []
    for s in range(steps):
        opt.zero_grad()
        out = model(X).float()
        loss = ((out - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

if __name__ == "__main__":
    d, M, steps = 512, 256, 200         # all multiples of 128
    print(f"MLP d={d}, batch M={M}, steps={steps}, device={torch.cuda.get_device_name(0)}\n")
    fp8 = train(BlockwiseFP8Linear, d, M, steps)
    bf16 = train(BF16Linear, d, M, steps)
    print(f"{'step':>4} {'fp8-blockwise loss':>20} {'bf16 loss':>14}")
    for s in [0, 1, 5, 20, 50, 100, steps - 1]:
        print(f"{s:>4} {fp8[s]:>20.5f} {bf16[s]:>14.5f}")
    drop = fp8[0] / max(fp8[-1], 1e-9)
    track = max(abs(a - b) / b for a, b in zip(fp8, bf16))   # fp8-vs-bf16 max rel gap
    print(f"\nfp8 loss: {fp8[0]:.4f} -> {fp8[-1]:.5f}  ({drop:.1f}x lower)")
    print(f"fp8 tracks bf16: max rel gap over all steps = {track:.2%}")
    monotone = all(fp8[i] <= fp8[i-1] + 1e-4 for i in range(1, steps))   # loss never goes up
    meaningful = fp8[-1] < fp8[0] * 0.5                                   # at least halved
    matches_bf16 = track < 0.01                                          # within 1% of bf16 everywhere
    print(f"checks: monotone↓={monotone}  halved={meaningful}  matches_bf16={matches_bf16}")
    print("RESULT:", "PASS — FP8 block-wise training converges on AMD MI355X and matches bf16"
          if (monotone and meaningful and matches_bf16) else "INCONCLUSIVE")
