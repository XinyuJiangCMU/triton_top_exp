#!/usr/bin/env python3
"""Wrapper: inject the blockwise-fp8 monkeypatch, then run Megatron's pretrain_gpt."""
import sys
sys.path.insert(0, "/root")                 # bw_fp8_gemm / grouped_fp8_gemm / te_*_patch
sys.path.insert(0, "/root/Megatron-LM")     # gpt_builders and other sibling modules
import te_moe_patch                   # applies dense + MoE blockwise patches at import
import runpy
runpy.run_path("/root/Megatron-LM/pretrain_gpt.py", run_name="__main__")
