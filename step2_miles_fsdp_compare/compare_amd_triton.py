#!/usr/bin/env python3
"""
Compare FSDP training-side dumps vs SGLang inference-side dumps
for AMD Triton true-on-policy verification.

Alignment strategy:
  - Inference side: pick the LAST decode step (shape[0]==1), skip prefill (shape[0]>1)
  - Training side : take the LAST token position (x[-1]) from the full-sequence tensor

The last inference decode step and the last training token correspond to the
same token (last response token of last sample), so their intermediate values
should be bit-wise identical under true-on-policy.
"""
from pathlib import Path
import re
import collections

import torch

# ===== Update these per run =====
SG_DIR = Path("/tmp/my_dumps/sglang_dump_1772731215.5847168")   # SGLang inference
TR_DIR = Path("/tmp/my_dumps/sglang_dump_1772731237.7235115")   # FSDP training

DIFF_THRESHOLD = 1e-3

# Names where we apply "last-token" alignment on the training side
# (i.e. training tensor shape is [seq_len, D] or [N])
ALIGN_LAST_TOKEN_NAMES = {
    "layer0_q_pre_norm", "layer0_k_pre_norm", "layer0_v_pre_norm",
    "layer0_q_post_norm", "layer0_k_post_norm",
    "layer0_q_post_rope", "layer0_k_post_rope",
    "layer0_attn_context_before_o_proj", "layer0_attn_out_after_o_proj",
    "q_pre_norm", "k_pre_norm", "v_pre_norm",
    "q_post_norm", "k_post_norm",
    "q_post_rope", "k_post_rope",
    "attn_context_before_o_proj", "attn_out_last_layer",
    # logprob / logit tensors: shape (N_response_tokens, ...) → take last
    "next_token_logprob_selected",
    "next_token_logits_raw",
    "next_token_logprobs_full",
    "next_token_id",
}

_PAT = re.compile(
    r"forward_pass_id=(\d+)___rank=(\d+)___name=(.*?)___dump_index=(\d+)\.pt$"
)


def scan(d: Path):
    """Scan dump dir → {name: [(fwd_id, dump_index, path)]} sorted by dump_index."""
    result = collections.defaultdict(list)
    for p in d.glob("*.pt"):
        m = _PAT.match(p.name)
        if not m:
            continue
        fwd, rank, name, idx = (
            int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        )
        if rank != 0:
            continue
        result[name].append((fwd, idx, p))
    for k in result:
        result[k].sort(key=lambda x: x[1])   # sort by dump_index
    return result


def load_pt(p: Path) -> torch.Tensor:
    x = torch.load(p, weights_only=False, map_location="cpu")
    if isinstance(x, dict) and "value" in x:
        x = x["value"]
    return x.float()


def pick_last_decode(entries):
    """
    From a list of (fwd_id, dump_index, path) from the INFERENCE side,
    return the last entry whose tensor has shape[0] == 1 (decode step, not prefill).
    Returns (fwd_id, dump_index, path, tensor) or None.
    """
    candidates = []
    for fwd, idx, p in entries:
        t = load_pt(p)
        first_dim = t.shape[0] if t.ndim >= 1 else 1
        if first_dim == 1:   # decode step
            candidates.append((fwd, idx, p, t))
    return candidates[-1] if candidates else None


def align_last_token(t: torch.Tensor) -> torch.Tensor:
    """Extract last token from training tensor."""
    if t.ndim == 2 and t.shape[0] > 1:
        return t[-1:]          # (1, D)
    if t.ndim == 1 and t.shape[0] > 1:
        return t[-1:]          # (1,)
    return t


def compare_name(name: str, sg_entries, tr_entries) -> None:
    # --- inference side: last decode step ---
    sg_result = pick_last_decode(sg_entries)
    if sg_result is None:
        print(f"[{name}] SKIP: no decode step found in inference")
        return
    sg_fwd, sg_idx, sg_path, sg_t = sg_result

    # --- training side: last entry, last token position ---
    tr_fwd, tr_idx, tr_path = tr_entries[-1]
    tr_t = load_pt(tr_path)

    if name in ALIGN_LAST_TOKEN_NAMES:
        tr_aligned = align_last_token(tr_t)
        sg_aligned = sg_t
    else:
        tr_aligned = tr_t
        sg_aligned = sg_t

    # Squeeze leading 1-dim on inference side if needed  (e.g. (1,D) vs (D,))
    if (
        tr_aligned.shape != sg_aligned.shape
        and sg_aligned.shape == tr_aligned.shape[1:]
        and tr_aligned.shape[0] == 1
    ):
        tr_aligned = tr_aligned.squeeze(0)

    print(
        f"[{name}]  "
        f"sg(fwd={sg_fwd},idx={sg_idx}) shape={tuple(sg_t.shape)}  "
        f"tr(fwd={tr_fwd},idx={tr_idx}) shape={tuple(tr_t.shape)}"
    )

    if tr_aligned.shape != sg_aligned.shape:
        print(
            f"  ⚠️  shape mismatch after align: "
            f"tr={tuple(tr_aligned.shape)} sg={tuple(sg_aligned.shape)}"
        )
        return

    diff = (tr_aligned - sg_aligned).abs()
    equal = torch.equal(tr_aligned, sg_aligned)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ok = "✅" if max_abs <= DIFF_THRESHOLD else "❌"
    print(
        f"  {ok}  equal={equal}  "
        f"max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}"
    )


def main() -> None:
    print(f"SG_DIR = {SG_DIR}")
    print(f"TR_DIR = {TR_DIR}\n")

    sg = scan(SG_DIR)
    tr = scan(TR_DIR)

    print(f"Inference unique names : {len(sg)}  (total files: {sum(len(v) for v in sg.values())})")
    print(f"Training  unique names : {len(tr)}  (total files: {sum(len(v) for v in tr.values())})")

    common = sorted(set(sg) & set(tr))
    print(f"Common names           : {len(common)}\n")

    for name in common:
        compare_name(name, sg[name], tr[name])


if __name__ == "__main__":
    main()
