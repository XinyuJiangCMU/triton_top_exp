#!/usr/bin/env python3
from pathlib import Path

import torch
# prepare_attn -- deterministic done.
# ====== 按你当前这轮实验日志更新 ======
# HF dump (day0 脚本侧): partial_name=1772518691.2323422
# SG dump (server 侧):   partial_name=1772518041.371588
HF_DIR = Path("/tmp/dumper/sglang_dump_1772518691.2323422")
SG_DIR = Path("/tmp/dumper/sglang_dump_1772518041.371588")


# ====== 按你的当前实验修改这些 index 映射 ======
HF_INDEX = {
    "layer0_hidden_in": 6,
    "layer0_attn_out": 7,
    "layer0_block_out": 8,
    "input_ids_for_compare": 1,
    "embedding_output": 2,
    "layer0_positions": 3,
    "layer0_attn_input_raw": 4,
    "layer0_attn_after_input_layernorm_only": 5,
    "layer0_attn_input_after_prepare": 5,
    "attn_input_last_layer": 9,
    "q_pre_norm": 10,
    "k_pre_norm": 11,
    "v_pre_norm": 12,
    "q_post_norm": -1,
    "k_post_norm": -1,
    "q_post_rope": -1,
    "k_post_rope": -1,
    "attn_context_before_o_proj": 13,
    "attn_out_last_layer": 14,
    "final_hidden_before_lm_head": 15,
    "lm_head_weight": 16,
}

SG_INDEX = {
    "layer0_hidden_in": 64,
    "layer0_q_pre_norm": 65,
    "layer0_k_pre_norm": 66,
    "layer0_v_pre_norm": 67,
    "layer0_attn_context_before_o_proj": 68,
    "layer0_attn_out": 69,
    "layer0_block_out": 70,
    "input_ids_for_compare": 58,
    "embedding_output": 59,
    "layer0_attn_input_raw": 60,
    "layer0_positions": 61,
    "layer0_attn_after_input_layernorm_only": 62,
    "layer0_attn_input_after_prepare": 63,
    "attn_input_last_layer": 71,
    "q_pre_norm": 72,
    "k_pre_norm": 73,
    "v_pre_norm": 74,
    "q_post_norm": 75,
    "k_post_norm": 76,
    "q_post_rope": 77,
    "k_post_rope": 78,
    "attn_context_before_o_proj": 79,
    "attn_out_last_layer": 80,
    "final_hidden_before_lm_head": 81,
    "lm_head_weight": -1,
}

# 允许同一个“逻辑对比名”在两侧使用不同的 dump 文件名
# 这里把 HF 的 layer0_attn_after_input_layernorm_only 映射到其现有命名。
HF_NAME_OVERRIDE = {
    "layer0_attn_after_input_layernorm_only": "layer0_attn_input_after_prepare",
}

# True: 将 hidden/attention 类张量统一对齐到“单步语义”
# - [B, T, D] -> [B, D]（取最后一步）
# - [T, D] -> [1, D]（取最后一个 token）
ALIGN_TO_SINGLE_STEP = True

ALIGN_NAMES = {
    "layer0_hidden_in",
    "layer0_attn_out",
    "layer0_block_out",
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
    "layer0_hidden_in",
    "layer0_attn_out",
    "layer0_block_out",
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    "layer0_attn_input_raw",
    "layer0_attn_after_input_layernorm_only",
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


def _extract_dump_index(p: Path) -> int:
    return int(p.stem.split("___dump_index=")[1])


def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    if idx > 0:
        p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    else:
        matches = sorted(
            d.glob(f"forward_pass_id=0___rank=0___name={name}___dump_index=*.pt"),
            key=_extract_dump_index,
        )
        if not matches:
            raise FileNotFoundError(f"missing dump for {name} under {d}")
        p = matches[0]
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


def compare(name: str) -> None:
    hf_idx = HF_INDEX[name]
    sg_idx = SG_INDEX[name]
    hf_name = HF_NAME_OVERRIDE.get(name, name)
    sg_name = name

    print(f"\n[{name}]")
    if hf_idx <= 0 or sg_idx <= 0:
        print(f"  -> skip (index not set): hf={hf_idx}, sg={sg_idx}")
        return

    try:
        x_hf = load_value(HF_DIR, hf_name, hf_idx)
        x_sg = load_value(SG_DIR, sg_name, sg_idx)
    except FileNotFoundError as e:
        print(f"  -> file missing, skip: {e}")
        return

    x_hf = normalize_for_compare(name, x_hf, side="hf")
    x_sg = normalize_for_compare(name, x_sg, side="sg")
    x_hf = align_single_step(name, x_hf)
    x_sg = align_single_step(name, x_sg)

    print("  hf shape/dtype:", tuple(x_hf.shape), x_hf.dtype)
    print("  sg shape/dtype:", tuple(x_sg.shape), x_sg.dtype)
    print("  shape_equal:", x_hf.shape == x_sg.shape)
    print("  dtype_equal:", x_hf.dtype == x_sg.dtype)

    if x_hf.shape != x_sg.shape:
        print("  -> shape mismatch, skip")
        return

    if x_hf.dtype == x_sg.dtype:
        print("  torch.equal:", torch.equal(x_hf, x_sg))
    else:
        print("  torch.equal: skip (dtype mismatch)")

    if x_hf.dtype.is_floating_point or x_sg.dtype.is_floating_point:
        diff = (x_hf.float() - x_sg.float()).abs()
        print("  value_equal_after_cast:", bool(torch.equal(x_hf.float(), x_sg.float())))
        print("  max_abs:", diff.max().item())
        print("  mean_abs:", diff.mean().item())
    else:
        neq = (x_hf != x_sg).sum().item()
        print("  value_equal_after_cast:", neq == 0)
        print("  neq_cnt:", neq)


def main() -> None:
    print("HF_DIR =", HF_DIR)
    print("SG_DIR =", SG_DIR)
    print("ALIGN_TO_SINGLE_STEP =", ALIGN_TO_SINGLE_STEP)

    for name in [
        "input_ids_for_compare",
        "embedding_output",
        "layer0_positions",
        "layer0_attn_input_raw",
        "layer0_attn_after_input_layernorm_only",
        "layer0_attn_input_after_prepare",
        "layer0_hidden_in",
        "layer0_attn_out",
        "layer0_block_out",
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
    ]:
        compare(name)


if __name__ == "__main__":
    main()
