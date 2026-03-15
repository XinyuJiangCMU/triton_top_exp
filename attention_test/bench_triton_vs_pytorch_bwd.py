"""Benchmark: Triton backward vs PyTorch native backward for causal attention.
Compares speed and peak memory usage.
"""
import torch
import time
import sys
sys.path.insert(0, "/app/sglang/python")
sys.path.insert(0, "/app/yuzhen1/miles")

from miles.backends.fsdp_utils.sglang_attn_bridge.triton_attn_bwd import triton_attention_backward
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd_unified


def pytorch_causal_attn_fwd_bwd(q, k, v, B, S, grad_out):
    """PyTorch native: materialize S×S attention, backward via autograd."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    D = q.shape[2]
    kv_group_num = num_heads // num_kv_heads

    q_r = q.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)

    q_4d = q_r.view(B, S, num_heads, D).transpose(1, 2).float()
    if kv_group_num > 1:
        k_exp = k_r.view(B, S, num_kv_heads, D).unsqueeze(3).expand(
            B, S, num_kv_heads, kv_group_num, D
        ).reshape(B, S, num_heads, D)
        v_exp = v_r.view(B, S, num_kv_heads, D).unsqueeze(3).expand(
            B, S, num_kv_heads, kv_group_num, D
        ).reshape(B, S, num_heads, D)
    else:
        k_exp = k_r.view(B, S, num_heads, D)
        v_exp = v_r.view(B, S, num_heads, D)

    k_4d = k_exp.transpose(1, 2).float()
    v_4d = v_exp.transpose(1, 2).float()

    scale = 1.0 / (D ** 0.5)
    attn = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
    attn.masked_fill_(causal_mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    o = torch.matmul(attn, v_4d)
    o = o.transpose(1, 2).reshape(B * S, num_heads, D)

    o.backward(grad_out.float())
    return q_r.grad, k_r.grad, v_r.grad


def triton_fwd_bwd(q, k, v, B, S, grad_out):
    """Triton forward + Triton backward."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    D = q.shape[2]
    kv_group_num = num_heads // num_kv_heads
    total = B * S
    device = q.device

    # Expand for forward
    if kv_group_num > 1:
        k_fwd = k.view(B, S, num_kv_heads, D).unsqueeze(3).expand(
            B, S, num_kv_heads, kv_group_num, D
        ).reshape(total, num_heads, D).contiguous()
        v_fwd = v.view(B, S, num_kv_heads, D).unsqueeze(3).expand(
            B, S, num_kv_heads, kv_group_num, D
        ).reshape(total, num_heads, D).contiguous()
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


def benchmark(fn, warmup=3, repeat=5):
    """Run fn with timing and peak memory measurement."""
    # Warmup
    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()

    # Timed runs
    times = []
    peak_mems = []
    for _ in range(repeat):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)

    return min(times), sum(peak_mems) / len(peak_mems)


device = "cuda"
configs = [
    # (B, S, num_heads, num_kv_heads, D)
    (1, 128, 16, 8, 128),
    (1, 256, 16, 8, 128),
    (1, 512, 16, 8, 128),
    (1, 1024, 16, 8, 128),
    (4, 256, 16, 8, 128),
    (4, 512, 16, 8, 128),
]

print(f"{'Config':<35} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'PT Mem (MB)':<15} {'Tri Mem (MB)':<15} {'Mem Save':<10}")
print("-" * 145)

for B, S, nh, nkv, D in configs:
    config_str = f"B={B}, S={S}, nh={nh}, nkv={nkv}"
    torch.manual_seed(42)
    q = torch.randn(B * S, nh, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B * S, nkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B * S, nkv, D, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(B * S, nh, D, device=device, dtype=torch.bfloat16)

    try:
        pt_time, pt_mem = benchmark(lambda: pytorch_causal_attn_fwd_bwd(q, k, v, B, S, grad_out))
        pt_str = f"{pt_time*1000:.2f}"
        pt_mem_str = f"{pt_mem:.1f}"
    except RuntimeError as e:
        if "out of memory" in str(e):
            pt_str = "OOM"
            pt_mem_str = "OOM"
            pt_time = float("inf")
            pt_mem = float("inf")
            torch.cuda.empty_cache()
        else:
            raise

    try:
        tri_time, tri_mem = benchmark(lambda: triton_fwd_bwd(q, k, v, B, S, grad_out))
        tri_str = f"{tri_time*1000:.2f}"
        tri_mem_str = f"{tri_mem:.1f}"
    except RuntimeError as e:
        if "out of memory" in str(e):
            tri_str = "OOM"
            tri_mem_str = "OOM"
            tri_time = float("inf")
            tri_mem = float("inf")
            torch.cuda.empty_cache()
        else:
            raise

    if pt_time != float("inf") and tri_time != float("inf"):
        speedup = f"{pt_time/tri_time:.2f}x"
        mem_save = f"{(1 - tri_mem/pt_mem)*100:.1f}%"
    else:
        speedup = "N/A"
        mem_save = "N/A"

    print(f"{config_str:<35} {pt_str:<15} {tri_str:<15} {speedup:<10} {pt_mem_str:<15} {tri_mem_str:<15} {mem_save:<10}")

    del q, k, v, grad_out
    torch.cuda.empty_cache()

print("\nDone.")
