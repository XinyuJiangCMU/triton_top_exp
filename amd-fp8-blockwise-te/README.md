# AMD FP8 block-wise training enablement (WIP)

Self-written Triton **block-scale FP8 GEMM** (DeepSeek 1×128 act / 128×128 weight, FP32 scales)
+ a python monkeypatch that makes **TE-native** `te.Linear` / `te.GroupedLinear` run
`Float8BlockScaling` on AMD MI355X **without recompiling TransformerEngine**.

## Why
ROCm/TransformerEngine rejects `NVTE_BLOCK_SCALING_1D/2D` in `rocm_gemm.cu` and in the
quantize dispatch; hipBLASLt has no `BLK128x128_32F` scale mode. The block-scale quantize
kernels already exist in-tree (gated); the only missing op is the **block-scale GEMM**.
So we wrote it in Triton (forward fp8 / backward bf16 — miles' FP8 training definition).

## Files
| file | what |
|---|---|
| `bw_fp8_gemm.py` | dense block-scale FP8 GEMM (Triton), M-masked for arbitrary token counts |
| `grouped_fp8_gemm.py` | variable-length **grouped** GEMM for MoE experts (single launch) |
| `te_blockwise_patch.py` | dense monkeypatch: gate + `tex.quantize` + `general_gemm` + `apply_normalization` |
| `te_moe_patch.py` | MoE monkeypatch: `tex.split_quantize` + `general_grouped_gemm` (fwd → grouped Triton) |
| `train_fp8_blockwise.py` | standalone fp8 Linear training proof (no TE) |
| `te_native_fp8_test.py` | TE-native `te.Linear` Float8BlockScaling fwd+bwd+train |
| `moe_fp8_test.py` | real MoE block (router + grouped experts) fp8 training |
| `pretrain_fp8.py` / `dryrun_fp8_moe.sh` | inject patch + 5-layer Qwen3-30B-A3B mock-data Megatron dry-run |
| `SUMMARY_for_Henry.md` | one-page report |

## Status
- ✅ standalone kernels verified (1.7e-3 vs fp8 ref, 3.7% quant error)
- ✅ dense `te.Linear` fp8 training converges (1.036→0.743), matches bf16
- ✅ MoE (router + grouped experts) fp8 training converges 27× (0.509→0.017)
- ⏳ real Megatron 5-layer Qwen3-30B-A3B dry-run: model builds, fp8 QKV path runs;
  debugging a Megatron config issue (`bias_dropout_add`)

## Boundaries
Forward fp8 via Triton; backward via bf16-dequant delegate. Native fp8 tensor-core MMA
is a perf follow-up. Kernels require K (and weight N) %128; token dim M arbitrary (masked).

Verified on `rlsys/miles:MI350-355-latest`, 1× MI355X (gfx950), TE 2.8.0+a365f2de.
