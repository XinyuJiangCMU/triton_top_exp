# HANDOFF — continue here (for the next agent)

You are picking up **FP8 block-wise training enablement on AMD MI355X**. Goal: make
DeepSeek-style block-scale FP8 training (the recipe DeepSeek-V3/V4 use) work on AMD,
since ROCm/TransformerEngine + hipBLASLt don't support it. Read this fully before touching anything.

## 1. Where things stand (what's DONE)
- **Gap is pinpointed**: the only missing op is the DeepSeek **block-scale FP8 GEMM**.
  `common/gemm/rocm_gemm.cu` only handles tensorwise(mode0) + MXFP8(mode1); it rejects
  `NVTE_BLOCK_SCALING_1D=2 / 2D=3` with `NVTE_ERROR("unsupported scaling mode")`. hipBLASLt
  has no `BLK128x128_32F` scale mode. The block-scale **quantize** kernels already exist
  in-tree but are gated by `pytorch/fp8.py check_fp8_block_scaling_support` (returns False on ROCm).
- **We wrote the kernel in Triton** (forward fp8 / backward bf16 = miles' FP8 definition):
  `bw_fp8_gemm.py` (dense), `grouped_fp8_gemm.py` (variable-length MoE grouped). Both numerically
  verified (1.7e-3 vs fp8 ref = bf16-output rounding; 3.7% fp8 quant error). Kernels are
  correctness-first: fp8→bf16 cast for the MMA (NOT native fp8 tensor-core yet).
- **Monkeypatch makes TE-native modules use them, no recompile**: `te_blockwise_patch.py` +
  `te_moe_patch.py`. Verified: `te.Linear` trains (loss 1.036→0.743), MoE router+experts trains
  (0.509→0.017, 27×).
- **Real Megatron dry-run STARTED but NOT finished** (see §4).

## 2. Environment / how to reproduce
- Container: `hai-te` (image `rlsys/miles:MI350-355-latest`, ROCm 7.0, gfx950/MI355X), runtime
  TE = `2.8.0+a365f2de` (ROCm/TransformerEngine release_v2.8_rocm).
- Files live in **container `/root/`** and host `/mnt/data/data/hai/` (same names). This repo
  folder is the snapshot.
- Run things: `docker exec hai-te bash -lc 'cd /root && HIP_VISIBLE_DEVICES=0 python3 <file>'`
  - `python3 bw_fp8_gemm.py` / `grouped_fp8_gemm.py` → kernel correctness
  - `python3 train_fp8_blockwise.py` → standalone fp8 training proof
  - `python3 te_native_fp8_test.py` → TE-native te.Linear fp8 (imports te_blockwise_patch)
  - `python3 moe_fp8_test.py` → MoE fp8 (imports te_moe_patch)
  - `bash dryrun_fp8_moe.sh` → the Megatron 5-layer Qwen3-30B-A3B mock dry-run (see §4)

## 3. The 5 monkeypatch points (TE rejects block scaling in 5 C++ entries)
In `te_blockwise_patch.py`: (1) gate `check_fp8_block_scaling_support`/`check_recipe_support`;
(2) `tex.quantize` (dense quantize — training path is `_QuantizeFunc.apply→tex.quantize`, NOT
update_quantized); (3) `general_gemm` (patched in `cpp_extensions.gemm` + `module.linear` +
`module.layernorm_linear` + `module.layernorm_mlp` — all import it by name); (4) `apply_normalization`
(fused RMSNorm+quantize for QKV/MLP → do bf16 norm then our quantize).
In `te_moe_patch.py`: (5) `tex.split_quantize` (grouped quantize) + `general_grouped_gemm`
(forward → grouped Triton; dgrad/wgrad → dequant-to-bf16 delegate).
Data is stashed on the TE tensor as `_bw_data` (fp8) / `_bw_scale` / `_bw_dim`.

## 4. THE IMMEDIATE NEXT TASK — finish the Megatron dry-run
`bash dryrun_fp8_moe.sh` currently: builds the 3.7B 5-layer model, the fp8 QKV+norm path runs,
then **fails in Megatron's `bias_dropout_add`** with a shape mismatch (`256` vs `65536` at dim 1;
256 = seq_length, 65536 = 256²). This is downstream of attention, NOT a kernel-math error.
- **Top hypothesis**: our dense `general_gemm` FORWARD branch returns a flat 2D `[M,N]` tensor
  (`bw_fp8_gemm(...)` output), but TE's original general_gemm returns an output whose leading
  dims match the input (so Megatron's residual add gets the right [S,B,H]). The attention output
  shape gets mangled → residual add fails.
- **First thing to try**: in `te_blockwise_patch._general_gemm`, temporarily DISABLE the Triton
  forward fast-path (just `if False:`) so the forward also takes the bf16-dequant delegate to
  ORIG general_gemm. If the Megatron dry-run then proceeds, the bug is confirmed to be the
  Triton forward path's output shape/contiguity → fix by reshaping `y` to match what ORIG returns
  (restore input leading dims) before returning. Same likely applies to the grouped forward in
  `te_moe_patch`.
- Other knobs already set in `dryrun_fp8_moe.sh`: `--no-bias-dropout-fusion`, `--no-rope-fusion`,
  `--no-masked-softmax-fusion`, `--no-gradient-accumulation-fusion`. If new torch.compile/Dynamo
  errors appear, suspect another fusion and disable it.

## 5. Roadmap (priority order)
1. **Finish the Megatron dry-run** (§4) → first real-stack proof that fp8 block-wise MoE trains.
   This is Henry's "random-weights dry-run" deliverable.
2. **Native fp8 tensor-core MMA** in both kernels (`tl.dot` on `tl.float8e4nv`) → perf; then
   benchmark vs bf16 to show the gap to hipBLASLt-if-it-existed.
3. **Backward fp8** (optional): currently backward is bf16 (matches miles). If full fp8 training
   is wanted, add fp8 dgrad/wgrad block-scale GEMMs (the gap table's 37 bwd symbols).
4. Scale to multi-GPU EP/TP for a bigger Qwen3-30B-A3B run; talk to ROCm/hipBLASLt in parallel
   about their block-scale plans (don't block on them — Henry: "wait for hipBLAS = game over").

## 6. Conventions (IMPORTANT)
- Git identity Xinyu Jiang <xinyuj2@andrew.cmu.edu>. Branches `wip/<desc>` or `pr/<desc>`.
- Commit via `git commit -F file.txt`. **NEVER add a `Co-Authored-By: Claude` trailer.**
  Always include `Co-authored-by: Xinyu Jiang` + `Co-authored-by: Zhiyao Jiang <jessicajiang324@gmail.com>`
  and a `Verification:` line (image / GPU / model).
- Read-only on other tenants' containers; don't restart other people's serving jobs.
- Background on the whole arc: `SUMMARY_for_Henry.md` (this folder) and the gap report
  `te-gap/te_gap_report.md` on the host.

## 7. One-line status to report up
"Kernel-level done (dense + MoE fp8 block-wise train & match bf16, all Triton, no recompile);
real Megatron 5-layer dry-run is one shape-bug away from running."
