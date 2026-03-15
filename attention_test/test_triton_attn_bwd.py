"""Correctness test: Triton backward vs PyTorch reference backward.

Covers various configurations:
  - Different sequence lengths (short S<=BLOCK, medium, long)
  - GQA (num_heads != num_kv_heads) and MHA (num_heads == num_kv_heads)
  - Different batch sizes
  - Different head dimensions
  - Edge cases: S not divisible by BLOCK_N (64)

For each config, compares dQ/dK/dV from Triton vs PyTorch reference using:
  - Max absolute diff
  - Mean absolute diff
  - Max relative diff
  - Cosine similarity
  - Inf/NaN check

Usage:
    PYTHONPATH="/app/sglang/python:/app/yuzhen1/miles" python test_triton_attn_bwd.py
"""

import torch
import sys

sys.path.insert(0, "/app/sglang/python")
sys.path.insert(0, "/app/yuzhen1/miles")

from miles.backends.fsdp_utils.sglang_attn_bridge.triton_attn_bwd import (
    triton_attention_backward,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd_unified,
)


def pytorch_reference_backward(q, k, v, B, S, grad_out):
    """PyTorch reference: materialize full S×S attention, autograd backward.

    This is the ground truth -- standard math, no tiling tricks.
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    D = q.shape[2]
    kv_group_num = num_heads // num_kv_heads

    q_r = q.detach().clone().float().requires_grad_(True)
    k_r = k.detach().clone().float().requires_grad_(True)
    v_r = v.detach().clone().float().requires_grad_(True)

    q_4d = q_r.view(B, S, num_heads, D).transpose(1, 2)

    if kv_group_num > 1:
        k_4d = (
            k_r.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(B, S, num_heads, D)
            .transpose(1, 2)
        )
        v_4d = (
            v_r.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(B, S, num_heads, D)
            .transpose(1, 2)
        )
    else:
        k_4d = k_r.view(B, S, num_heads, D).transpose(1, 2)
        v_4d = v_r.view(B, S, num_heads, D).transpose(1, 2)

    scale = 1.0 / (D**0.5)
    attn = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
    attn.masked_fill_(causal_mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    o = torch.matmul(attn, v_4d).transpose(1, 2).reshape(B * S, num_heads, D)

    o.backward(grad_out.float())
    return q_r.grad, k_r.grad, v_r.grad


def triton_forward_backward(q, k, v, B, S, grad_out):
    """Triton forward (extend_attention_fwd_unified) + Triton backward."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    D = q.shape[2]
    kv_group_num = num_heads // num_kv_heads
    total = B * S
    device = q.device

    if kv_group_num > 1:
        k_fwd = (
            k.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(total, num_heads, D)
            .contiguous()
        )
        v_fwd = (
            v.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(total, num_heads, D)
            .contiguous()
        )
    else:
        k_fwd = k.contiguous()
        v_fwd = v.contiguous()

    o = torch.empty_like(q)
    qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q, o, k_fwd, v_fwd,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        max_len_extend=S, is_causal=True,
    )

    dq, dk, dv = triton_attention_backward(q, k, v, o, grad_out, B, S)
    return dq, dk, dv


def compare(name, tri, ref):
    """Compare triton result vs reference, return (pass, stats_str)."""
    tri_f = tri.float()
    ref_f = ref.float()

    has_inf = tri_f.isinf().any().item()
    has_nan = tri_f.isnan().any().item()

    abs_diff = (tri_f - ref_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    denom = ref_f.abs().clamp(min=1e-6)
    max_rel = (abs_diff / denom).max().item()
    mean_rel = (abs_diff / denom).mean().item()

    cos = torch.nn.functional.cosine_similarity(
        tri_f.reshape(-1), ref_f.reshape(-1), dim=0
    ).item()

    # For near-zero tensors (e.g. S=1 dQ), cosine similarity is undefined.
    # Fall back to checking that absolute errors are tiny.
    if ref_f.norm() < 1e-4:
        ok = (not has_inf) and (not has_nan) and (max_abs < 0.01)
    else:
        ok = (not has_inf) and (not has_nan) and (cos > 0.999)
    stats = (
        f"{name}: cos={cos:.6f}, max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"mean_rel={mean_rel*100:.2f}%, "
        f"inf={has_inf}, nan={has_nan}"
    )
    return ok, stats


def run_test(B, S, num_heads, num_kv_heads, D, seed=42):
    """Run one test configuration."""
    config = f"B={B}, S={S}, nh={num_heads}, nkv={num_kv_heads}, D={D}"

    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(B * S, num_heads, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B * S, num_kv_heads, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B * S, num_kv_heads, D, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(B * S, num_heads, D, device=device, dtype=torch.bfloat16)

    # Reference
    dq_ref, dk_ref, dv_ref = pytorch_reference_backward(q, k, v, B, S, grad_out)

    # Triton
    dq_tri, dk_tri, dv_tri = triton_forward_backward(q, k, v, B, S, grad_out)

    ok_q, s_q = compare("dQ", dq_tri, dq_ref)
    ok_k, s_k = compare("dK", dk_tri, dk_ref)
    ok_v, s_v = compare("dV", dv_tri, dv_ref)

    passed = ok_q and ok_k and ok_v
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {config}")
    print(f"  {s_q}")
    print(f"  {s_k}")
    print(f"  {s_v}")

    del q, k, v, grad_out
    torch.cuda.empty_cache()
    return passed


# ── Test Configurations ──
# Each tuple: (B, S, num_heads, num_kv_heads, D, description)
tests = [
    # --- Sequence length coverage ---
    # S <= BLOCK_N (64): single block, trivial case
    (1, 1, 16, 8, 128, "S=1, minimal"),
    (1, 32, 16, 8, 128, "S=32, within one block"),
    (1, 64, 16, 8, 128, "S=64, exactly one block"),
    # S > BLOCK_N: multi-block, tests LSE correctness
    (1, 65, 16, 8, 128, "S=65, just over one block (edge)"),
    (1, 128, 16, 8, 128, "S=128, two blocks"),
    (1, 256, 16, 8, 128, "S=256, four blocks"),
    (1, 512, 16, 8, 128, "S=512, eight blocks"),
    (1, 1024, 16, 8, 128, "S=1024, realistic training length"),
    # S not divisible by BLOCK_N
    (1, 100, 16, 8, 128, "S=100, not aligned to block"),
    (1, 200, 16, 8, 128, "S=200, not aligned to block"),

    # --- GQA vs MHA ---
    (1, 256, 16, 16, 128, "MHA: nh==nkv (no GQA)"),
    (1, 256, 16, 8, 128, "GQA: 2x group"),
    (1, 256, 16, 4, 128, "GQA: 4x group"),
    (1, 256, 32, 8, 128, "GQA: 4x group, more heads"),

    # --- Batch size ---
    (2, 256, 16, 8, 128, "B=2"),
    (4, 128, 16, 8, 128, "B=4"),
    (8, 64, 16, 8, 128, "B=8, many short seqs"),

    # --- Head dimension ---
    (1, 256, 16, 8, 64, "D=64, smaller head dim"),
    (1, 256, 16, 8, 128, "D=128, standard"),

    # --- Combined stress ---
    (4, 512, 32, 8, 128, "stress: B=4, S=512, nh=32, GQA 4x"),
]

print("=" * 70)
print("Triton Backward Correctness Test vs PyTorch Reference")
print("=" * 70)

total = 0
passed = 0
for B, S, nh, nkv, D, desc in tests:
    total += 1
    print(f"\n--- Test {total}: {desc} ---")
    try:
        if run_test(B, S, nh, nkv, D):
            passed += 1
    except Exception as e:
        print(f"\n[ERROR] B={B}, S={S}, nh={nh}, nkv={nkv}, D={D}: {e}")

print("\n" + "=" * 70)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {total - passed}")
print("=" * 70)
