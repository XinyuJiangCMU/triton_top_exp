#!/usr/bin/env python3
"""A real MoE block (top-1 router + grouped experts) training in FP8 block-wise on AMD."""
import torch, torch.nn as nn, torch.nn.functional as F
import te_moe_patch                                   # dense + moe blockwise patches
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling

BLK, dev = 128, "cuda"
recipe = Float8BlockScaling()

class MoE(nn.Module):
    def __init__(self, d, E, dff):
        super().__init__()
        self.E, self.d, self.dff = E, d, dff
        self.router = nn.Linear(d, E).to(dev).to(torch.bfloat16)
        self.fc1 = te.GroupedLinear(E, d, dff, bias=False, params_dtype=torch.bfloat16, device=dev)
        self.fc2 = te.GroupedLinear(E, dff, d, bias=False, params_dtype=torch.bfloat16, device=dev)

    def forward(self, x):                              # x[M,d]
        M = x.shape[0]
        probs = F.softmax(self.router(x).float(), dim=-1)
        top1 = probs.argmax(-1)                        # [M] expert id per token
        gate = probs.gather(1, top1[:, None]).squeeze(1).to(torch.bfloat16)
        # dropless routing: real per-expert token counts (no capacity padding;
        # the Triton kernels now mask arbitrary M)
        order = torch.argsort(top1)
        m_splits = [int((top1 == e).sum()) for e in range(self.E)]
        xin = x[order]
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):  # experts in FP8 block-wise
            h = self.fc2(F.relu(self.fc1(xin, m_splits)), m_splits)
        out = torch.zeros(M, self.d, device=dev, dtype=torch.bfloat16)
        out = out.index_copy(0, order, h)
        return out * gate[:, None]

torch.manual_seed(0)
d, E, dff, M = 512, 4, 512, 512
moe = MoE(d, E, dff)
torch.manual_seed(123)
X = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
Wt = torch.randn(d, d, device=dev, dtype=torch.bfloat16) * (d ** -0.5)
Y = F.relu(X @ Wt.T).float()
opt = torch.optim.Adam(moe.parameters(), lr=1e-2)

losses = []
for s in range(300):
    opt.zero_grad()
    out = moe(X).float()
    loss = ((out - Y) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(loss.item())

# routing balance on last step
with torch.no_grad():
    top1 = F.softmax(moe.router(X).float(), -1).argmax(-1)
    counts = [int((top1 == e).sum()) for e in range(E)]
print(f"MoE: d={d} experts={E} dff={dff} tokens={M}, device={torch.cuda.get_device_name(0)}")
print(f"routing counts per expert (last step): {counts}")
print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}   steps: " +
      ", ".join(f"{i}:{losses[i]:.3f}" for i in [0, 10, 50, 100, 149]))
trend = sum(losses[-10:]) / 10 < sum(losses[:10]) / 10        # clear downward trend
print(f"loss reduction: {losses[0] / losses[-1]:.1f}x")
print("RESULT:", "PASS — FP8 block-wise MoE (router + grouped experts) trains on AMD MI355X"
      if (losses[-1] < losses[0] * 0.1 and trend) else "CHECK")
