#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import torch

DUMPER_ROOT = Path("/tmp/dumper")

NAME_ORDER = [
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    "layer0_attn_input_raw",
    "layer0_attn_input_after_prepare",
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
    "lm_head_weight",
]

# True: 将 hidden/attention 类张量统一对齐到“单步语义”
# - [B, T, D] -> [B, D]（取最后一步）
# - [T, D] -> [1, D]（取最后一个 token）
ALIGN_TO_SINGLE_STEP = True

ALIGN_NAMES = {
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
}

HF_DROP_LAST_TOKEN_NAMES = {
    # HF teacher-forcing 输入常包含“最后一个 response token”，对齐 SGLang prefill 时需去掉
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
}

SQUEEZE_BATCH1_NAMES = {
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    "layer0_attn_input_raw",
    "layer0_attn_input_after_prepare",
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
}

FILE_RE = re.compile(
    r"^forward_pass_id=0___rank=0___name=(?P<name>.+)___dump_index=(?P<index>\d+)\.pt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF and SGLang dumper tensors around the attention path."
    )
    parser.add_argument("--hf-dir", type=Path, default=None)
    parser.add_argument("--sg-dir", type=Path, default=None)
    parser.add_argument(
        "--focus",
        nargs="*",
        default=[
            "q_post_norm",
            "k_post_norm",
            "q_post_rope",
            "k_post_rope",
            "attn_context_before_o_proj",
        ],
        help="Names to compare. Defaults to the 5 key attention checkpoints.",
    )
    parser.add_argument(
        "--list-latest",
        action="store_true",
        help="Print available dump directories under /tmp/dumper and exit.",
    )
    return parser.parse_args()


def list_dump_dirs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.glob("sglang_dump_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )


def discover_index_map(dump_dir: Path) -> dict[str, int]:
    index_map: dict[str, int] = {}
    for path in dump_dir.glob("forward_pass_id=0___rank=0___name=*___dump_index=*.pt"):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        name = match.group("name")
        index = int(match.group("index"))
        index_map[name] = index
    return index_map


def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    obj = torch.load(p, weights_only=False, map_location="cpu")
    return obj["value"] if isinstance(obj, dict) and "value" in obj else obj


def align_single_step(name: str, x: torch.Tensor) -> torch.Tensor:
    if not ALIGN_TO_SINGLE_STEP or name not in ALIGN_NAMES:
        return x

    if x.ndim == 3:
        return x[:, -1, :]
    if x.ndim == 2 and x.shape[0] > 1:
        return x[-1:, :]
    return x


def normalize_for_compare(name: str, x: torch.Tensor, side: str) -> torch.Tensor:
    # 把 [1, ...] 统一成 [...]
    if name in SQUEEZE_BATCH1_NAMES and x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    # HF teacher-forcing 序列通常比 SGLang prefill 多一个末尾 token
    if side == "hf" and name in HF_DROP_LAST_TOKEN_NAMES:
        if x.ndim == 1 and x.shape[0] > 1:
            x = x[:-1]
        elif x.ndim == 2 and x.shape[0] > 1:
            x = x[:-1, :]

    return x


def squeeze_single_step_tail(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    return x


def compare(name: str, hf_dir: Path, sg_dir: Path, hf_index: dict[str, int], sg_index: dict[str, int]) -> None:
    hf_idx = hf_index.get(name, -1)
    sg_idx = sg_index.get(name, -1)

    print(f"\n[{name}]")
    if hf_idx <= 0 or sg_idx <= 0:
        print(f"  -> skip (index not set): hf={hf_idx}, sg={sg_idx}")
        return

    try:
        x_hf = load_value(hf_dir, name, hf_idx)
        x_sg = load_value(sg_dir, name, sg_idx)
    except FileNotFoundError as e:
        print(f"  -> file missing, skip: {e}")
        return

    x_hf = normalize_for_compare(name, x_hf, side="hf")
    x_sg = normalize_for_compare(name, x_sg, side="sg")
    x_hf = align_single_step(name, x_hf)
    x_sg = align_single_step(name, x_sg)
    x_hf = squeeze_single_step_tail(x_hf)
    x_sg = squeeze_single_step_tail(x_sg)

    print("  hf shape/dtype:", tuple(x_hf.shape), x_hf.dtype)
    print("  sg shape/dtype:", tuple(x_sg.shape), x_sg.dtype)

    if x_hf.shape != x_sg.shape:
        print("  -> shape mismatch, skip")
        return

    if x_hf.dtype == x_sg.dtype:
        print("  torch.equal:", torch.equal(x_hf, x_sg))
    else:
        print("  torch.equal: False (dtype mismatch)")

    if x_hf.dtype.is_floating_point or x_sg.dtype.is_floating_point:
        diff = (x_hf.float() - x_sg.float()).abs()
        print("  max_abs:", diff.max().item())
        print("  mean_abs:", diff.mean().item())
    else:
        neq = (x_hf != x_sg).sum().item()
        print("  neq_cnt:", neq)


def main() -> None:
    args = parse_args()
    dump_dirs = list_dump_dirs(DUMPER_ROOT)
    if args.list_latest:
        for dump_dir in dump_dirs:
            print(dump_dir)
        return

    hf_dir = args.hf_dir or (dump_dirs[-1] if dump_dirs else None)
    sg_dir = args.sg_dir or (dump_dirs[-2] if len(dump_dirs) >= 2 else None)
    if hf_dir is None or sg_dir is None:
        raise SystemExit("Need at least two dump directories or pass --hf-dir and --sg-dir explicitly.")

    hf_index = discover_index_map(hf_dir)
    sg_index = discover_index_map(sg_dir)

    print("HF_DIR =", hf_dir)
    print("SG_DIR =", sg_dir)
    print("ALIGN_TO_SINGLE_STEP =", ALIGN_TO_SINGLE_STEP)
    print("FOCUS =", args.focus)
    print("HF_INDEX =", {name: hf_index.get(name, -1) for name in NAME_ORDER})
    print("SG_INDEX =", {name: sg_index.get(name, -1) for name in NAME_ORDER})

    for name in args.focus:
        compare(name, hf_dir, sg_dir, hf_index, sg_index)


if __name__ == "__main__":
    main()
