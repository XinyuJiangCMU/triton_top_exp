#!/usr/bin/env python3
"""Verify whether HF/Qwen3 RoPE can reproduce SGLang dumped RoPE outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use SGLang dumped pre-rope inputs and post-rope outputs as ground truth, "
            "then compare SGLang local rerun and HF/Qwen3 rope outputs against the dump."
        )
    )
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--q-in-name", type=str, default="q_post_norm")
    parser.add_argument("--k-in-name", type=str, default="k_post_norm")
    parser.add_argument("--q-out-name", type=str, default="q_post_rope")
    parser.add_argument("--k-out-name", type=str, default="k_post_rope")
    parser.add_argument("--positions-name", type=str, default="layer0_positions")
    parser.add_argument("--q-in-index", type=int, default=None)
    parser.add_argument("--k-in-index", type=int, default=None)
    parser.add_argument("--q-out-index", type=int, default=None)
    parser.add_argument("--k-out-index", type=int, default=None)
    parser.add_argument("--positions-index", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_unique_dump_file(dump_dir: Path, name: str, index: int | None) -> Path:
    if index is not None:
        path = dump_dir / f"forward_pass_id=0___rank=0___name={name}___dump_index={index}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing dump file: {path}")
        return path

    matches = sorted(
        path
        for path in dump_dir.glob(f"forward_pass_id=0___rank=0___name={name}___dump_index=*.pt")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"No dump file matched name={name} under {dump_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple dump files matched name={name} under {dump_dir}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def resolve_positions_file(dump_dir: Path, name: str, index: int | None) -> tuple[str, Path]:
    candidates = [name]
    if index is None and name == "layer0_positions":
        candidates.append("position_ids")

    errors: list[str] = []
    for candidate in candidates:
        try:
            return candidate, resolve_unique_dump_file(dump_dir, candidate, index)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")

    raise FileNotFoundError("Failed to resolve positions dump file. " + " | ".join(errors))


def load_dump_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=False, map_location="cpu")
    value = obj["value"] if isinstance(obj, dict) and "value" in obj else obj
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Dump {path} did not contain a tensor, got {type(value).__name__}")
    return value


def print_tensor_meta(label: str, value: torch.Tensor) -> None:
    print(
        f"{label}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
        f"stride={tuple(value.stride())} contiguous={value.is_contiguous()}"
    )


def canonicalize_hidden_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2:
        # [T, hidden] or [1, hidden] -> [1, T, hidden]
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:
        # [hidden] -> [1, 1, hidden]
        return tensor.unsqueeze(0).unsqueeze(0)
    raise ValueError(
        f"{name} expected [B, T, hidden], [T, hidden], [1, hidden], or [hidden], got shape={tuple(tensor.shape)}"
    )


def canonicalize_positions(name: str, positions: torch.Tensor, target_bt: tuple[int, int]) -> torch.Tensor:
    batch, seq = target_bt
    if positions.ndim == 2 and positions.shape == (batch, seq + 1):
        positions = positions[:, :-1]
    elif positions.ndim == 1 and positions.shape[0] == seq + 1:
        positions = positions[:-1]

    if positions.ndim == 2 and positions.shape == (batch, seq):
        return positions
    if positions.ndim == 1 and positions.shape[0] == seq:
        return positions.unsqueeze(0)
    if positions.ndim == 2 and positions.shape == (1, seq):
        return positions
    if positions.ndim == 1 and positions.shape[0] == 1 and seq == 1:
        return positions.unsqueeze(0)
    raise ValueError(
        f"{name} shape={tuple(positions.shape)} not aligned with target (B, T)=({batch}, {seq})"
    )


def infer_layout(name: str, tensor: torch.Tensor, config: Any) -> tuple[int, int]:
    if tensor.ndim != 3:
        raise ValueError(f"{name} expected canonical rank-3 [B, T, hidden], got shape={tuple(tensor.shape)}")
    hidden = tensor.shape[-1]
    head_dim = config.hidden_size // config.num_attention_heads
    if hidden % head_dim != 0:
        raise ValueError(f"{name} hidden={hidden} not divisible by head_dim={head_dim}")
    num_heads = hidden // head_dim
    return num_heads, head_dim


def reshape_for_hf(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    # [B, T, H*D] -> [B, H, T, D]
    return x.view(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()


def flatten_from_hf(x: torch.Tensor) -> torch.Tensor:
    # [B, H, T, D] -> [B, T, H*D]
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1).contiguous()


def compare_pair(label: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float | None, float | None, bool]:
    print(f"\n[{label}]")
    print("  lhs shape/dtype:", tuple(lhs.shape), lhs.dtype)
    print("  rhs shape/dtype:", tuple(rhs.shape), rhs.dtype)
    if lhs.shape != rhs.shape:
        print("  -> shape mismatch")
        return None, None, False
    diff = (lhs.float() - rhs.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    bitwise_equal = lhs.dtype == rhs.dtype and torch.equal(lhs.contiguous(), rhs.contiguous())
    print("  max_abs:", max_abs)
    print("  mean_abs:", mean_abs)
    print("  bitwise_equal:", bitwise_equal)
    return max_abs, mean_abs, bitwise_equal


def build_sglang_rope_with_device(
    *,
    get_rope_fn,
    device: torch.device,
    head_dim: int,
    config: Any,
    dtype: torch.dtype,
):
    previous_device = None
    if hasattr(torch, "get_default_device"):
        previous_device = torch.get_default_device()
    torch.set_default_device(device.type if device.index is None else f"{device.type}:{device.index}")
    try:
        return get_rope_fn(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            dtype=dtype,
        )
    finally:
        if previous_device is not None:
            torch.set_default_device(previous_device)


def main() -> None:
    args = parse_args()

    os.environ["SGLANG_USE_AITER"] = "0"
    if "/app/sglang/python" not in sys.path:
        sys.path.insert(0, "/app/sglang/python")

    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )
    from sglang.srt.server_args import set_global_server_args_for_scheduler
    from sglang.srt.layers.rotary_embedding import get_rope

    q_in_path = resolve_unique_dump_file(args.dump_dir, args.q_in_name, args.q_in_index)
    k_in_path = resolve_unique_dump_file(args.dump_dir, args.k_in_name, args.k_in_index)
    q_out_path = resolve_unique_dump_file(args.dump_dir, args.q_out_name, args.q_out_index)
    k_out_path = resolve_unique_dump_file(args.dump_dir, args.k_out_name, args.k_out_index)
    positions_name, positions_path = resolve_positions_file(
        args.dump_dir, args.positions_name, args.positions_index
    )

    q_in = load_dump_tensor(q_in_path)
    k_in = load_dump_tensor(k_in_path)
    q_out_dump = load_dump_tensor(q_out_path)
    k_out_dump = load_dump_tensor(k_out_path)
    positions = load_dump_tensor(positions_path)

    if args.verbose:
        print_tensor_meta("q_in_dump_raw", q_in)
        print_tensor_meta("k_in_dump_raw", k_in)
        print_tensor_meta("q_out_dump_raw", q_out_dump)
        print_tensor_meta("k_out_dump_raw", k_out_dump)
        print_tensor_meta("positions_raw", positions)

    q_in = canonicalize_hidden_tensor(args.q_in_name, q_in)
    k_in = canonicalize_hidden_tensor(args.k_in_name, k_in)
    q_out_dump = canonicalize_hidden_tensor(args.q_out_name, q_out_dump)
    k_out_dump = canonicalize_hidden_tensor(args.k_out_name, k_out_dump)

    if q_in.shape[:2] != k_in.shape[:2] or q_in.shape[:2] != q_out_dump.shape[:2] or q_in.shape[:2] != k_out_dump.shape[:2]:
        raise ValueError(
            "Canonicalized q/k input/output shapes disagree: "
            f"q_in={tuple(q_in.shape)}, k_in={tuple(k_in.shape)}, "
            f"q_out={tuple(q_out_dump.shape)}, k_out={tuple(k_out_dump.shape)}"
        )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    q_heads, head_dim = infer_layout(args.q_in_name, q_in, config)
    k_heads, k_head_dim = infer_layout(args.k_in_name, k_in, config)
    if k_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim}, k={k_head_dim}")

    positions = canonicalize_positions(positions_name, positions, q_in.shape[:2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_in = q_in.contiguous().to(device)
    k_in = k_in.contiguous().to(device)
    q_out_dump = q_out_dump.contiguous().to(device)
    k_out_dump = k_out_dump.contiguous().to(device)
    positions = positions.contiguous().to(device)

    # SGLang get_rope() only needs this flag to choose the rl_on_policy cache-init path.
    set_global_server_args_for_scheduler(
        SimpleNamespace(rl_on_policy_target="fsdp" if device.type == "cuda" else None)
    )

    if args.verbose:
        print("SGLang rope import: sglang.srt.layers.rotary_embedding.get_rope")
        print(
            "HF rope imports: transformers.models.qwen3.modeling_qwen3."
            "Qwen3RotaryEmbedding, apply_rotary_pos_emb"
        )
        print(f"positions_source={positions_name}")
        print(f"execution_device={device}")
        print(f"inferred q_heads={q_heads}, k_heads={k_heads}, head_dim={head_dim}")
        print("shape semantics:")
        print("  dump q/k in: [B, T, H*D]")
        print("  HF rope in: [B, H, T, D]")
        print("  SGLang rope in: [B, T, H*D]")
        print("  cos/sin source: generated from positions via each implementation")
        print_tensor_meta("q_in_dump", q_in)
        print_tensor_meta("k_in_dump", k_in)
        print_tensor_meta("q_out_dump", q_out_dump)
        print_tensor_meta("k_out_dump", k_out_dump)
        print_tensor_meta("positions", positions)

    # Compare 1 inputs/outputs against SGLang local rerun.
    sglang_rope = build_sglang_rope_with_device(
        get_rope_fn=get_rope,
        device=device,
        head_dim=head_dim,
        config=config,
        dtype=q_in.dtype,
    )
    q_out_sg, k_out_sg = sglang_rope(positions.clone(), q_in.clone(), k_in.clone())

    # Compare 2: HF/Qwen3 rope on the same SGLang dumped inputs.
    hf_rope = Qwen3RotaryEmbedding(config=config)
    q_hf_in = reshape_for_hf(q_in, q_heads, head_dim)
    k_hf_in = reshape_for_hf(k_in, k_heads, head_dim)
    cos, sin = hf_rope(q_hf_in, positions)
    q_hf_heads, k_hf_heads = apply_rotary_pos_emb(q_hf_in, k_hf_in, cos, sin)
    q_out_hf = flatten_from_hf(q_hf_heads).to(q_in.dtype)
    k_out_hf = flatten_from_hf(k_hf_heads).to(k_in.dtype)

    if args.verbose:
        print_tensor_meta("q_hf_in", q_hf_in)
        print_tensor_meta("k_hf_in", k_hf_in)
        print_tensor_meta("cos", cos)
        print_tensor_meta("sin", sin)
        print_tensor_meta("q_out_sg_local", q_out_sg)
        print_tensor_meta("k_out_sg_local", k_out_sg)
        print_tensor_meta("q_out_hf", q_out_hf)
        print_tensor_meta("k_out_hf", k_out_hf)

    # Compare 1: SG dump vs SG local rerun
    q_sg_max, q_sg_mean, q_sg_bit = compare_pair("sg_dump_vs_sg_local_q", q_out_dump.cpu(), q_out_sg.cpu())
    k_sg_max, k_sg_mean, k_sg_bit = compare_pair("sg_dump_vs_sg_local_k", k_out_dump.cpu(), k_out_sg.cpu())

    # Compare 2: SG dump vs HF output
    q_hf_max, q_hf_mean, q_hf_bit = compare_pair("sg_dump_vs_hf_q", q_out_dump.cpu(), q_out_hf.cpu())
    k_hf_max, k_hf_mean, k_hf_bit = compare_pair("sg_dump_vs_hf_k", k_out_dump.cpu(), k_out_hf.cpu())

    # Compare 3: delta compare
    compare_pair("delta_q", (q_out_dump - q_in).cpu(), (q_out_hf - q_in).cpu())
    compare_pair("delta_k", (k_out_dump - k_in).cpu(), (k_out_hf - k_in).cpu())

    sg_match_dump = (
        q_sg_max == 0.0
        and k_sg_max == 0.0
        and q_sg_mean == 0.0
        and k_sg_mean == 0.0
        and q_sg_bit
        and k_sg_bit
    )
    hf_match_dump = (
        q_hf_max == 0.0
        and k_hf_max == 0.0
        and q_hf_mean == 0.0
        and k_hf_mean == 0.0
        and q_hf_bit
        and k_hf_bit
    )

    print()
    print(f"SG_RERUN_MATCH_DUMP: {'YES' if sg_match_dump else 'NO'}")
    print(f"HF_MATCH_SGLANG_DUMP: {'YES' if hf_match_dump else 'NO'}")
    if sg_match_dump and hf_match_dump:
        print("Conclusion: rope implementations and positions semantics appear aligned for this dumped input.")
    elif sg_match_dump and not hf_match_dump:
        print("Conclusion: local SGLang rerun matches the dump, but HF does not; this points to HF rope or positions/cos/sin semantics.")
    elif not sg_match_dump:
        print("Conclusion: local SGLang rerun does not reproduce the dump yet; shape or positions semantics are still not fully aligned.")
    else:
        print("Conclusion: unable to make a clean call; re-check shapes, positions source, and device path.")


if __name__ == "__main__":
    main()
