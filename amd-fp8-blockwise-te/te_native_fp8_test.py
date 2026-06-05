#!/usr/bin/env python3
"""TE-native te.Linear under Float8BlockScaling on ROCm, via the monkeypatch."""
import torch
import te_blockwise_patch                      # applies the patch on import
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling

torch.manual_seed(0)
dev = "cuda"
M, K, N = 256, 512, 512                         # all %128
recipe = Float8BlockScaling()

def build():
    torch.manual_seed(1)
    return te.Linear(K, N, bias=False, params_dtype=torch.bfloat16, device=dev)

# ---- 1) single fwd+bwd, TE-native fp8 path ----
lin = build()
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = lin(x)
y.sum().backward()
print(f"[1] TE-native fp8 fwd+bwd OK: y{tuple(y.shape)} {y.dtype}, "
      f"x.grad={'ok' if x.grad is not None else 'None'}, "
      f"w.grad={'ok' if lin.weight.grad is not None else 'None'}")

# ---- 2) numerical sanity vs pure bf16 Linear (same weights) ----
linb = build()
xb = x.detach().clone().requires_grad_(True)
yb = xb @ linb.weight.detach().T
rel = ((y.detach().float() - yb.float()).norm() / yb.float().norm()).item()
print(f"[2] fwd rel-err vs bf16 reference = {rel:.3e}  (fp8 quant error, expect ~1e-2)")

# ---- 3) train a tiny MLP of TE Linears under fp8_autocast ----
torch.manual_seed(2)
model = te.LayerNormMLP if False else None
mlp = torch.nn.Sequential(build(), torch.nn.ReLU(), build()).to(dev)
torch.manual_seed(123)
X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
Wt = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * (K ** -0.5)
Y = (X @ Wt.T).float()
opt = torch.optim.SGD(mlp.parameters(), lr=0.5)
losses = []
for s in range(200):
    opt.zero_grad()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = mlp(X)
    loss = ((out.float() - Y) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print(f"[3] TE-native fp8 training: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")
print("    steps: " + ", ".join(f"{i}:{losses[i]:.3f}" for i in [0, 10, 50, 100, 199]))
monotone = all(losses[i] <= losses[i-1] + 2e-3 for i in range(1, len(losses)))
print("RESULT:", "PASS — TE-native te.Linear trains under Float8BlockScaling on AMD MI355X"
      if (losses[-1] < losses[0] * 0.8 and monotone) else "CHECK")
