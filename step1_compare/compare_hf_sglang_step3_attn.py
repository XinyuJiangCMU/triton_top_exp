#!/usr/bin/env python3
from pathlib import Path
import re

import torch
# ====== 按你当前这轮实验日志更新 ======
# HF dump (day0 脚本侧): partial_name=1772517893.9605722
# SG dump (server 侧):   partial_name=1772517881.2454364
HF_DIR = Path("/tmp/dumper/sglang_dump_1772517893.9605722")
SG_DIR = Path("/tmp/dumper/sglang_dump_1772517881.2454364")

# ====== 按你的当前实验修改这些 index 映射 ======
HF_INDEX = {
    # 本轮 HF 未 dump 这两项
    "input_ids_for_compare": -1,
    "embedding_output": -1,
    "layer0_positions": 1,
    "layer0_attn_input_raw": 2,
    # 本轮 HF 未单独 dump 该点，避免语义误对齐，直接 skip
    "layer0_attn_after_input_layernorm_only": -1,
    "layer0_attn_input_after_prepare": 3,
    "layer0_hidden_component_after_postprocess": 12,
    "layer0_decoder_output_full": 13,
    "layer8_attn_input_after_prepare": -1,
    "layer8_hidden_component_after_postprocess": -1,
    "layer8_decoder_output_full": -1,
    "layer16_attn_input_after_prepare": -1,
    "layer16_hidden_component_after_postprocess": -1,
    "layer16_decoder_output_full": -1,
    "layer24_attn_input_after_prepare": -1,
    "layer24_hidden_component_after_postprocess": -1,
    "layer24_decoder_output_full": -1,
    "layer32_attn_input_after_prepare": -1,
    "layer32_hidden_component_after_postprocess": -1,
    "layer32_decoder_output_full": -1,
    "attn_input_last_layer": 14,
    "q_pre_norm": 15,
    "k_pre_norm": 16,
    "v_pre_norm": 17,
    # 本轮 HF 未产出这 4 个点，设为 -1 自动跳过
    "q_post_norm": -1,
    "k_post_norm": -1,
    "q_post_rope": -1,
    "k_post_rope": -1,
    "attn_context_before_o_proj": 18,
    "attn_out_last_layer": 19,
    "final_hidden_before_lm_head": 20,
    "lm_head_weight": 21,
    # 新增 layer0 细粒度点（后续可直接打开 compare）
    "layer0_attn_context_before_o_proj": 4,
    "layer0_attn_out_after_o_proj": 5,
    "layer0_q_pre_norm": 6,
    "layer0_k_pre_norm": 7,
    "layer0_v_pre_norm": 8,
    "layer0_q_post_norm": -1,
    "layer0_k_post_norm": -1,
    "layer0_q_post_rope": -1,
    "layer0_k_post_rope": -1,
    "layer0_after_attn_residual_add": 9,
    "layer0_post_attention_layernorm_output": 10,
    "layer0_mlp_output": 11,
}

SG_INDEX = {
    "input_ids_for_compare": 1,
    "embedding_output": 2,
    "layer0_attn_input_raw": 3,
    "layer0_positions": 4,
    "layer0_attn_after_input_layernorm_only": 5,
    "layer0_attn_input_after_prepare": 6,
    # SG 里 layer0_attn_input_after_prepare 有重复 index（6/7），取第一处
    "layer0_hidden_component_after_postprocess": 20,
    "layer0_decoder_output_full": 21,
    "layer8_attn_input_after_prepare": -1,
    "layer8_hidden_component_after_postprocess": -1,
    "layer8_decoder_output_full": -1,
    "layer16_attn_input_after_prepare": -1,
    "layer16_hidden_component_after_postprocess": -1,
    "layer16_decoder_output_full": -1,
    "layer24_attn_input_after_prepare": -1,
    "layer24_hidden_component_after_postprocess": -1,
    "layer24_decoder_output_full": -1,
    "layer32_attn_input_after_prepare": -1,
    "layer32_hidden_component_after_postprocess": -1,
    "layer32_decoder_output_full": -1,
    "attn_input_last_layer": 22,
    "q_pre_norm": 23,
    "k_pre_norm": 24,
    "v_pre_norm": 25,
    "q_post_norm": 26,
    "k_post_norm": 27,
    "q_post_rope": 28,
    "k_post_rope": 29,
    "attn_context_before_o_proj": 30,
    "attn_out_last_layer": 31,
    "final_hidden_before_lm_head": 32,
    "lm_head_weight": 33,
    # 新增 layer0 细粒度点（后续可直接打开 compare）
    "layer0_q_pre_norm": 8,
    "layer0_k_pre_norm": 9,
    "layer0_v_pre_norm": 10,
    "layer0_q_post_norm": 11,
    "layer0_k_post_norm": 12,
    "layer0_q_post_rope": 13,
    "layer0_k_post_rope": 14,
    "layer0_attn_context_before_o_proj": 15,
    "layer0_attn_out_after_o_proj": 16,
    "layer0_after_attn_residual_add": 17,
    "layer0_post_attention_layernorm_output": 18,
    "layer0_mlp_output": 19,
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
    "layer0_attn_input_after_prepare",
    "layer0_attn_context_before_o_proj",
    "layer0_attn_out_after_o_proj",
    "layer0_q_pre_norm",
    "layer0_k_pre_norm",
    "layer0_v_pre_norm",
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "layer0_after_attn_residual_add",
    "layer0_post_attention_layernorm_output",
    "layer0_mlp_output",
    "layer0_hidden_component_after_postprocess",
    "layer0_decoder_output_full",
    "layer8_attn_input_after_prepare",
    "layer8_hidden_component_after_postprocess",
    "layer8_decoder_output_full",
    "layer16_attn_input_after_prepare",
    "layer16_hidden_component_after_postprocess",
    "layer16_decoder_output_full",
    "layer24_attn_input_after_prepare",
    "layer24_hidden_component_after_postprocess",
    "layer24_decoder_output_full",
    "layer32_attn_input_after_prepare",
    "layer32_hidden_component_after_postprocess",
    "layer32_decoder_output_full",
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
    "layer0_attn_after_input_layernorm_only",
    "layer0_attn_input_after_prepare",
    "layer0_attn_context_before_o_proj",
    "layer0_attn_out_after_o_proj",
    "layer0_q_pre_norm",
    "layer0_k_pre_norm",
    "layer0_v_pre_norm",
    "layer0_q_post_norm",
    "layer0_k_post_norm",
    "layer0_q_post_rope",
    "layer0_k_post_rope",
    "layer0_after_attn_residual_add",
    "layer0_post_attention_layernorm_output",
    "layer0_mlp_output",
    "layer0_hidden_component_after_postprocess",
    "layer0_decoder_output_full",
    "layer8_attn_input_after_prepare",
    "layer8_hidden_component_after_postprocess",
    "layer8_decoder_output_full",
    "layer16_attn_input_after_prepare",
    "layer16_hidden_component_after_postprocess",
    "layer16_decoder_output_full",
    "layer24_attn_input_after_prepare",
    "layer24_hidden_component_after_postprocess",
    "layer24_decoder_output_full",
    "layer32_attn_input_after_prepare",
    "layer32_hidden_component_after_postprocess",
    "layer32_decoder_output_full",
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


def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    obj = torch.load(p, weights_only=False, map_location="cpu")
    return obj["value"] if isinstance(obj, dict) and "value" in obj else obj


def resolve_index(d: Path, name: str, idx: int) -> int:
    if idx > 0:
        return idx
    pattern = f"forward_pass_id=0___rank=0___name={name}___dump_index=*.pt"
    candidates = list(d.glob(pattern))
    if not candidates:
        return idx
    best = -1
    for p in candidates:
        m = re.search(r"dump_index=(\d+)\.pt$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best if best > 0 else idx


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
    hf_idx = resolve_index(HF_DIR, HF_NAME_OVERRIDE.get(name, name), HF_INDEX[name])
    sg_idx = resolve_index(SG_DIR, name, SG_INDEX[name])
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
    print("HF_DIR =", HF_DIR)
    print("SG_DIR =", SG_DIR)
    print("ALIGN_TO_SINGLE_STEP =", ALIGN_TO_SINGLE_STEP)

    for name in [
        # "input_ids_for_compare",
        # "embedding_output",
        # ---------------- layer0 inner ----------------
        "layer0_positions",
        "layer0_attn_input_raw",
        "layer0_attn_after_input_layernorm_only",
        "layer0_attn_input_after_prepare",
        "layer0_q_pre_norm",
        "layer0_k_pre_norm",
        "layer0_v_pre_norm",
        "layer0_q_post_norm",
        "layer0_k_post_norm",
        "layer0_q_post_rope",
        "layer0_k_post_rope",
        "layer0_attn_context_before_o_proj",
        "layer0_attn_out_after_o_proj",
        "layer0_after_attn_residual_add",
        "layer0_post_attention_layernorm_output",
        "layer0_mlp_output",
        "layer0_hidden_component_after_postprocess",
        "layer0_decoder_output_full",
        # "layer8_attn_input_after_prepare",
        # "layer8_hidden_component_after_postprocess",
        # "layer8_decoder_output_full",
        # "layer16_attn_input_after_prepare",
        # "layer16_hidden_component_after_postprocess",
        # "layer16_decoder_output_full",
        # "layer24_attn_input_after_prepare",
        # "layer24_hidden_component_after_postprocess",
        # "layer24_decoder_output_full",
        # "layer32_attn_input_after_prepare",
        # "layer32_hidden_component_after_postprocess",
        # "layer32_decoder_output_full",
        # ---------------- last layer ----------------
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