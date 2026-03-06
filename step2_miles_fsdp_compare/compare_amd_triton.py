#!/usr/bin/env python3
"""
Compare SG (SGLang inference) vs TR (FSDP training) dumps for
true-on-policy verification.

Designed for --rollout-max-response-len 1 (single token per response).

How it works
============
SG side (inference):
  - Prefill: processes prompt tokens → intermediates shape (P, dim)
  - Produces logits at last position → predicts the 1 response token
  - May also have decode steps (shape 1) and re-computation passes

TR side (training):
  - Teacher-forcing on full padded sequence → intermediates shape (seq_len, dim)
  - Produces logits for response token(s) → shape (N_resp, vocab)

Comparison strategy
===================
1. PREFILL intermediates:  SG_prefill[:P] vs TR[:P]
   Both sides process the same prompt tokens with the same weights,
   so these should be identical under true-on-policy.

2. LOGITS: SG's first-prefill logit vs TR's first logit (row 0)
   The prefill's logit predicts the first response token.

Dump pairing
============
We find the first SG PREFILL forward pass (intermediates with shape[0] > 1)
and pair its logit with TR's first logit. This avoids the bug where the first
prefill was not captured by the dumper and decode-step logits got incorrectly
matched to TR teacher-forcing logits.

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
RUN_DIR = Path(os.environ.get(
    "COMPARE_RUN_DIR",
    "/app/true_on_policy/results/miles_fsdp_debug_v2",
))
DUMPS_DIR = RUN_DIR / "dumps"
DIFF_THRESHOLD = 1e-3

# ─── File name pattern ───
PAT = re.compile(
    r"forward_pass_id=(\d+)___rank=(\d+)___name=(.*?)___dump_index=(\d+)\.pt$"
)

LOGIT_NAMES = {
    "next_token_id",
    "next_token_logits_raw",
    "next_token_logprobs_full",
    "next_token_logprob_selected",
}


# ─── Helpers ───

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
    """Auto-discover latest SG + TR dump pair.
    SG dumps contain input_ids_for_compare; TR dumps do not."""
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


def report(a: torch.Tensor, b: torch.Tensor, label: str = "") -> bool:
    """Print comparison.  Returns True if within threshold."""
    if a.dtype != b.dtype:
        print(f"  {label}⚠️  dtype mismatch: {a.dtype} vs {b.dtype}")
        return False
    eq = torch.equal(a, b)
    if a.is_floating_point():
        d = (a - b).abs()
        mx, mn = d.max().item(), d.mean().item()
        ok = mx <= DIFF_THRESHOLD
        tag = "✅" if ok else "❌"
        print(f"  {label}{tag}  equal={eq}  max={mx:.3e}  mean={mn:.3e}  dtype={a.dtype}")
        return ok
    else:
        neq = int((a != b).sum())
        ok = neq == 0
        tag = "✅" if ok else "❌"
        print(f"  {label}{tag}  equal={eq}  neq={neq}/{a.numel()}  dtype={a.dtype}")
        return ok


def find_first_prefill_idx(sg: dict) -> int:
    """Find the dump_index of SG's first prefill intermediate (shape[0] > 1).

    Returns the dump_index, or None if not found.
    """
    probe_name = "layer0_q_pre_norm"
    if probe_name not in sg:
        return None
    for idx, p in sg[probe_name]:
        t = load(p)
        if t.ndim >= 2 and t.shape[0] > 1:
            return idx
    return None


def get_sg_logit_for_prefill(entries: list, prefill_idx: int) -> tuple:
    """Get the logit entry whose dump_index is the first one >= prefill_idx.

    The prefill forward pass dumps intermediates first (idx ~77-100),
    then logits right after (idx ~102-105). So the first logit entry
    with dump_index >= prefill_idx is the correct one.
    """
    for idx, p in entries:
        if idx >= prefill_idx:
            return (idx, p)
    return None


# ─── Main ───

def main():
    sg_dir, tr_dir = find_pair()
    print(f"SG_DIR = {sg_dir}")
    print(f"TR_DIR = {tr_dir}\n")

    sg = scan(sg_dir)
    tr = scan(tr_dir)
    common = sorted(set(sg) & set(tr))

    print(f"SG names: {len(sg)}  ({sum(len(v) for v in sg.values())} files)")
    print(f"TR names: {len(tr)}  ({sum(len(v) for v in tr.values())} files)")
    print(f"Common  : {len(common)}")

    # ── Find the first prefill forward pass in SG ──
    prefill_idx = find_first_prefill_idx(sg)
    if prefill_idx is not None:
        print(f"\nSG first prefill starts at dump_index={prefill_idx}")
    else:
        print("\n⚠️  WARNING: No prefill found in SG dump (all shape[0]==1).")
        print("   The first prefill may not have been captured by the dumper.")
        print("   Falling back to first entry for each name.\n")

    # ── Dump structure table ──
    print(f"\n{'Name':<45} {'SG':>4}  shapes" + " " * 30 + f"{'TR':>4}  shapes")
    print("-" * 110)
    for name in common:
        sg_shapes = sorted({tuple(load(p).shape) for _, p in sg[name]})
        tr_shapes = sorted({tuple(load(p).shape) for _, p in tr[name]})
        sg_str = "  ".join(str(s) for s in sg_shapes)
        tr_str = "  ".join(str(s) for s in tr_shapes)
        print(f"  {name:<43} {len(sg[name]):>4}  {sg_str:<40} {len(tr[name]):>4}  {tr_str}")

    # ================================================================
    # 1. Logit / Token ID comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("1. Logit / Token ID (first prefill pass)")
    print("=" * 60)

    for name in sorted(LOGIT_NAMES & set(common)):
        # SG: use the logit from the first prefill forward pass
        if prefill_idx is not None:
            entry = get_sg_logit_for_prefill(sg[name], prefill_idx)
            if entry is None:
                entry = sg[name][0]  # fallback
            sg_idx, sg_path = entry
        else:
            sg_idx, sg_path = sg[name][0]

        sg_t = load(sg_path)

        # TR: first entry
        tr_idx, tr_path = tr[name][0]
        tr_t = load(tr_path)

        print(f"\n[{name}]  sg_idx={sg_idx}  tr_idx={tr_idx}")
        print(f"  SG shape={tuple(sg_t.shape)}  TR shape={tuple(tr_t.shape)}")

        # TR may have N_resp > 1; take first row only for 1:1 comparison
        if tr_t.ndim >= 1 and tr_t.shape[0] > sg_t.shape[0]:
            tr_t = tr_t[:sg_t.shape[0]]
            print(f"  (TR sliced to first {sg_t.shape[0]} row(s) for comparison)")

        if name == "next_token_id":
            print(f"  SG={sg_t.tolist()}  TR={tr_t.tolist()}")

        report(sg_t, tr_t)

    # ================================================================
    # 2. Attention intermediates — prefill comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("2. Attention intermediates (SG prefill[:P] vs TR[:P])")
    print("=" * 60)

    attn_names = sorted(set(common) - LOGIT_NAMES)

    # Sanity check: print raw values for layer0_q_pre_norm
    sanity_name = "layer0_q_pre_norm"
    if sanity_name in sg and sanity_name in tr:
        sg_pf = None
        for idx, p in sg[sanity_name]:
            t = load(p)
            if t.ndim >= 2 and t.shape[0] > 1:
                sg_pf = t
                break
        if sg_pf is not None:
            tr_t = load(tr[sanity_name][0][1])
            P = min(sg_pf.shape[0], tr_t.shape[0])
            D = min(8, sg_pf.shape[-1])
            print(f"\n[sanity check: {sanity_name}]  P={P}")
            print(f"  (if values are close → same sample, precision diff)")
            print(f"  (if values are totally different → wrong sample pairing)")
            for tok in range(min(3, P)):
                sv = "  ".join(f"{v:+.4f}" for v in sg_pf[tok, :D].float().tolist())
                tv = "  ".join(f"{v:+.4f}" for v in tr_t[tok, :D].float().tolist())
                dv = "  ".join(
                    f"{v:+.4f}"
                    for v in (sg_pf[tok, :D].float() - tr_t[tok, :D].float()).tolist()
                )
                print(f"  tok[{tok}] SG= {sv}")
                print(f"  tok[{tok}] TR= {tv}")
                print(f"  tok[{tok}] Δ = {dv}")

    for name in attn_names:
        # SG: find first prefill (shape[0] > 1)
        sg_prefill = None
        sg_idx = None
        for idx, p in sg[name]:
            t = load(p)
            if t.ndim >= 2 and t.shape[0] > 1:
                sg_prefill = t
                sg_idx = idx
                break

        if sg_prefill is None:
            print(f"\n[{name}]  SKIP (no prefill in SG)")
            continue

        # TR: first entry
        tr_idx, tr_path = tr[name][0]
        tr_t = load(tr_path)

        P = min(sg_prefill.shape[0], tr_t.shape[0])

        print(
            f"\n[{name}]  sg_idx={sg_idx} ({tuple(sg_prefill.shape)})  "
            f"tr_idx={tr_idx} ({tuple(tr_t.shape)})  P={P}"
        )
        report(sg_prefill[:P], tr_t[:P])


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
