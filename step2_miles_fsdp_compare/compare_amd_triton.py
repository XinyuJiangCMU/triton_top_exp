#!/usr/bin/env python3
"""
Compare SG (SGLang inference) vs TR (FSDP training) dumps.
Goal: bitwise-identical logprob for the selected token.

Usage:
  python compare_amd_triton.py
  COMPARE_RUN_DIR=/path/to/results python compare_amd_triton.py
"""
import os
import re
import sys
import datetime
import collections
from pathlib import Path

import torch

# ─── Config ───
DUMPS_DIR = Path(os.environ.get(
    "COMPARE_DUMPS_DIR",
    "/app/true_on_policy/results/amd_triton_dumps",
))

PAT = re.compile(
    r"forward_pass_id=(\d+)___rank=(\d+)___name=(.*?)___dump_index=(\d+)\.pt$"
)

# Skip these — not useful for comparison
SKIP_NAMES = {
    "next_token_logits_raw",       # raw logits, noisy, don't affect training
    "next_token_logprobs_full",    # full vocab logprobs, same issue
    "input_ids_for_compare",       # SG-only
    "embedding_output",            # SG-only
    "lm_head_weight",              # SG-only
    "final_hidden_before_lm_head", # SG-only
}

# Display raw tensor names directly (e.g. layer0_q_pre_norm).
DISPLAY_NAMES = {}

DISPLAY_GROUPS = [
    (
        "== Layer0 / Prepare ==",
        [
            "layer0_attn_input_raw",
            "layer0_attn_after_input_layernorm_only",
            "layer0_attn_input_after_prepare",
            "layer0_hidden_in",
            "layer0_residual",
        ],
    ),
    (
        "== Layer0 / QKV+Norm+RoPE+Attn ==",
        [
            "layer0_q_pre_norm",
            "layer0_k_pre_norm",
            "layer0_v_pre_norm",
            "layer0_q_norm_input_native",
            "layer0_k_norm_input_native",
            "layer0_q_post_norm_native",
            "layer0_k_post_norm_native",
            "layer0_q_post_norm",
            "layer0_k_post_norm",
            "layer0_q_post_rope",
            "layer0_k_post_rope",
            "layer0_attn_context_before_o_proj",
            "layer0_attn_out",
            "layer0_attn_out_after_o_proj",
        ],
    ),
    (
        "== Layer0 / Block ==",
        [
            "layer0_block_out_before_residual_add",
            "layer0_block_out_after_residual_add",
            "layer0_block_out",
        ],
    ),
    (
        "== Layer1 / Prepare+Attn+Block ==",
        [
            "layer1_attn_input_raw",
            "layer1_attn_after_input_layernorm_only",
            "layer1_attn_input_after_prepare",
            "layer1_hidden_in",
            "layer1_residual",
            "layer1_q_pre_norm",
            "layer1_k_pre_norm",
            "layer1_v_pre_norm",
            "layer1_q_post_norm",
            "layer1_k_post_norm",
            "layer1_q_post_rope",
            "layer1_k_post_rope",
            "layer1_attn_context_before_o_proj",
            "layer1_attn_out",
            "layer1_attn_out_after_o_proj",
            "layer1_block_out_before_residual_add",
            "layer1_block_out_after_residual_add",
            "layer1_block_out",
        ],
    ),
    (
        "== Last Layer / QKV+Norm+RoPE+Attn ==",
        [
            "attn_input_last_layer",
            "q_pre_norm",
            "k_pre_norm",
            "v_pre_norm",
            "q_norm_input_native",
            "k_norm_input_native",
            "q_post_norm_native",
            "k_post_norm_native",
            "q_post_norm",
            "k_post_norm",
            "q_post_rope",
            "k_post_rope",
            "attn_context_before_o_proj",
            "attn_out_last_layer",
        ],
    ),
    (
        "== Tokens / Logprob ==",
        [
            "next_token_id",
            "next_token_logprob_selected",
        ],
    ),
]


def load(p: Path) -> torch.Tensor:
    x = torch.load(p, weights_only=False, map_location="cpu")
    if isinstance(x, dict) and "value" in x:
        x = x["value"]
    return x


def scan(d: Path) -> dict:
    """Returns {name: [(dump_index, path)]} sorted by dump_index, rank=0 only."""
    out = collections.defaultdict(list)
    for p in d.glob("*.pt"):
        m = PAT.match(p.name)
        if m and int(m.group(2)) == 0:
            out[m.group(3)].append((int(m.group(4)), p))
    for k in out:
        out[k].sort()
    return out


def find_pair():
    """Auto-discover latest SG + TR dump pair."""
    dirs = sorted(
        [p for p in DUMPS_DIR.glob("sglang_dump_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    sg = tr = None
    for d in dirs:
        has_input_ids = any(d.glob("*___name=input_ids_for_compare___*.pt"))
        if has_input_ids and sg is None:
            sg = d
        elif not has_input_ids and tr is None:
            tr = d
        if sg and tr:
            break
    if not sg or not tr:
        raise RuntimeError(f"Need both SG and TR dumps under {DUMPS_DIR}")
    return sg, tr


def find_first_prefill_idx(sg: dict) -> int:
    """Find dump_index of SG's first prefill (shape[0] > 1)."""
    for probe_name in ["layer0_q_pre_norm", "layer0_q_post_rope", "layer0_v_pre_norm"]:
        if probe_name not in sg:
            continue
        for idx, p in sg[probe_name]:
            t = load(p)
            if t.ndim >= 2 and t.shape[0] > 1:
                return idx
    return None




def _align_non_logit_shapes(sg_t: torch.Tensor, tr_t: torch.Tensor):
    """Best-effort shape alignment for compare without changing semantics."""
    if sg_t.shape == tr_t.shape:
        return sg_t, tr_t

    # Common case: one side is transposed [feat, seq] vs [seq, feat]
    if sg_t.ndim == 2 and tr_t.ndim == 2 and sg_t.shape == tr_t.T.shape:
        return sg_t, tr_t.T

    # Fallback for non-comparable shapes: return as-is, caller will report/skip
    return sg_t, tr_t


def get_sg_entry(entries: list, prefill_idx: int, is_logit: bool) -> tuple:
    """Pick the right SG entry.
    - For logits: first entry with dump_index >= prefill_idx (from the prefill pass)
    - For intermediates: first entry with shape[0] > 1 (the prefill itself)
    """
    if is_logit and prefill_idx is not None:
        for idx, p in entries:
            if idx >= prefill_idx:
                return (idx, p)
    return entries[0] if entries else None


def build_grouped_order(common: set[str]):
    grouped = []
    used = set()
    for title, names in DISPLAY_GROUPS:
        present = [n for n in names if n in common]
        if present:
            grouped.append((title, present))
            used.update(present)

    extra = sorted(common - used)
    if extra:
        grouped.append(("== Other ==", extra))
    return grouped


def main():
    sg_dir, tr_dir = find_pair()
    print(f"SG = {sg_dir.name}")
    print(f"TR = {tr_dir.name}\n")

    sg = scan(sg_dir)
    tr = scan(tr_dir)

    prefill_idx = find_first_prefill_idx(sg)

    common = set(sg) & set(tr) - SKIP_NAMES
    grouped_order = build_grouped_order(common)

    if not grouped_order:
        print("ERROR: no common tensor names to compare")
        return

    for title, names in grouped_order:
        print(title)
        for name in names:
            display = DISPLAY_NAMES.get(name, name)
            is_logit = name.startswith("next_token_")

            # Pick SG tensor
            if is_logit:
                sg_idx, sg_path = get_sg_entry(sg[name], prefill_idx, is_logit=True)
                sg_t = load(sg_path)
            else:
                # Find first prefill entry (shape[0] > 1)
                sg_t = None
                for idx, p in sg[name]:
                    t = load(p)
                    if t.ndim >= 2 and t.shape[0] > 1:
                        sg_t = t
                        break
                if sg_t is None:
                    continue  # no prefill data

            # Pick TR tensor
            tr_t = load(tr[name][0][1])

            # Align shapes
            if is_logit and tr_t.ndim >= 1 and tr_t.shape[0] > sg_t.shape[0]:
                tr_t = tr_t[:sg_t.shape[0]]
            elif not is_logit:
                P = min(sg_t.shape[0], tr_t.shape[0])
                sg_t = sg_t[:P]
                tr_t = tr_t[:P]
                sg_t, tr_t = _align_non_logit_shapes(sg_t, tr_t)

            if sg_t.shape != tr_t.shape:
                print(f"❌ {display}  shape_mismatch SG={tuple(sg_t.shape)} TR={tuple(tr_t.shape)}  "
                      f"[{str(sg_t.dtype).replace('torch.', '')} vs {str(tr_t.dtype).replace('torch.', '')}]")
                continue

            # Compare
            eq = torch.equal(sg_t, tr_t)
            tag = "✅" if eq else "❌"

            def short_dtype(t):
                return str(t.dtype).replace("torch.", "")

            dtypes = f"  [{short_dtype(sg_t)} vs {short_dtype(tr_t)}]"

            if eq:
                print(f"{tag} {display}{dtypes}")
            elif name == "next_token_id":
                print(f"{tag} {display}  SG={sg_t.tolist()}  TR={tr_t.tolist()}{dtypes}")
            elif sg_t.is_floating_point():
                d = (sg_t.float() - tr_t.float()).abs()
                print(f"{tag} {display}  max={d.max().item():.3e}  mean={d.mean().item():.3e}{dtypes}")
            else:
                neq = int((sg_t != tr_t).sum())
                print(f"{tag} {display}  mismatched={neq}/{sg_t.numel()}{dtypes}")
        print()


if __name__ == "__main__":
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sg_dir, tr_dir = find_pair()
    sg_tag = sg_dir.name.replace("sglang_dump_", "sg")
    tr_tag = tr_dir.name.replace("sglang_dump_", "tr")
    out_path = RESULTS_DIR / f"{ts}__{sg_tag}__{tr_tag}.txt"

    class Tee:
        def __init__(self, *s):
            self.s = s
        def write(self, d):
            for s in self.s:
                s.write(d)
        def flush(self):
            for s in self.s:
                s.flush()

    with open(out_path, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nResults saved to: {out_path}")
