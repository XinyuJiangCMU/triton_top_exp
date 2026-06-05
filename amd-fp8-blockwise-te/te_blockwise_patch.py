"""
Lightweight monkeypatch: make TE-native te.pytorch.Linear run Float8BlockScaling
(DeepSeek 1x128 act / 128x128 weight) on ROCm — no C++ recompile.

TE's ROCm path is broken in two spots:
  - quantize: cast_kernels_hip.cuh rejects block scaling   -> we patch Float8BlockQuantizer.update_quantized
  - gemm   : rocm_gemm.cu rejects block scaling             -> we patch general_gemm

Strategy: do the blockwise fp8 quantization ourselves (torch), stash fp8 data + scales
on the TE tensor; in general_gemm, dequant blockwise operands back to bf16 and hand them
to TE's WORKING bf16 GEMM (so all TN/NN/NT layouts + transpose are handled by TE itself).
The forward GEMM additionally routes to our Triton block-scale fp8 kernel.

Import this module (after bw_fp8_gemm is importable) BEFORE building the Linear.
"""
import torch
import transformer_engine.pytorch.fp8 as tefp8
import transformer_engine.pytorch.cpp_extensions.gemm as tegemm
import transformer_engine.pytorch.module.linear as telinear
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer, Float8BlockwiseQTensorBase,
)
import transformer_engine_torch as tex          # available after TE init above
from bw_fp8_gemm import E4M3_MAX, BLK, bw_fp8_gemm

# ---- 1) open the ROCm gates ----
tefp8.check_fp8_block_scaling_support = lambda: (True, "")
tefp8.check_recipe_support = lambda r: None

# ---- 2) quantize: compute fp8 + scale in torch, stash on the tensor ----
def _quant(src2d, dim):
    M, K = src2d.shape
    if dim == 1:                                   # 1x128 (activation / grad)
        b = src2d.reshape(M, K // BLK, BLK).float()
        sc = (b.abs().amax(2, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
        q = (b / sc).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(M, K)
        return q, sc.squeeze(-1).contiguous()                      # [M, K/128]
    b = src2d.reshape(M // BLK, BLK, K // BLK, BLK).float()         # 128x128 (weight)
    sc = (b.abs().amax((1, 3), keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    q = (b / sc).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(M, K)
    return q, sc.reshape(M // BLK, K // BLK).contiguous()           # [M/128, K/128]

def _fill(quantizer, src, dst):
    s = src.contiguous()
    s2 = s.reshape(-1, s.shape[-1])
    M, K = s2.shape
    assert K % BLK == 0, f"blockwise patch needs K %128, got K={K}"
    assert quantizer.block_scaling_dim == 1 or M % BLK == 0, f"2D weight needs M %128, got M={M}"
    q, sc = _quant(s2, quantizer.block_scaling_dim)
    dst._bw_data, dst._bw_scale = q, sc
    dst._bw_dim, dst._bw_shape = quantizer.block_scaling_dim, tuple(s.shape)
    dst._rowwise_data, dst._fp8_dtype = q, quantizer.dtype
    return dst

def _update_quantized(self, src, dst, *, noop_flag=None):
    return _fill(self, src, dst)
Float8BlockQuantizer.update_quantized = _update_quantized

# the real training path goes _QuantizeFunc.apply -> tex.quantize(tensor, quantizer);
# intercept tex.quantize for block quantizers (handles 2-arg and 4-arg forms)
_ORIG_TEX_QUANTIZE = tex.quantize
def _tex_quantize(*a, **k):
    tensor, quantizer = a[0], a[1]
    if isinstance(quantizer, Float8BlockQuantizer):
        dst = a[2] if len(a) >= 3 else quantizer.make_empty(
            tensor.shape, dtype=tensor.dtype, device=tensor.device)
        return _fill(quantizer, tensor, dst)
    return _ORIG_TEX_QUANTIZE(*a, **k)
tex.quantize = _tex_quantize

# ---- 3) gemm: dequant blockwise operands -> bf16, delegate to TE's working gemm;
#         the forward (TN, both blockwise) goes through our Triton kernel ----
_ORIG_GEMM = tegemm.general_gemm

def _dequant_bf16(t):
    q = t._bw_data.float(); sc = t._bw_scale; M, K = q.shape
    if t._bw_dim == 1:
        d = (q.reshape(M, K // BLK, BLK) * sc[:, :, None]).reshape(M, K)
    else:
        d = (q.reshape(M // BLK, BLK, K // BLK, BLK) * sc[:, None, :, None]).reshape(M, K)
    return d.to(torch.bfloat16)

def _general_gemm(A, B, *args, **kwargs):
    a_bw = isinstance(A, Float8BlockwiseQTensorBase) and hasattr(A, "_bw_data")
    b_bw = isinstance(B, Float8BlockwiseQTensorBase) and hasattr(B, "_bw_data")
    if not (a_bw or b_bw):
        return _ORIG_GEMM(A, B, *args, **kwargs)

    layout = kwargs.get("layout", "TN")
    # Forward: general_gemm(weight[N,K] dim2, input[M,K] dim1, layout=TN) -> input @ weight^T
    if a_bw and b_bw and layout == "TN" and A._bw_dim == 2 and B._bw_dim == 1:
        out_dtype = kwargs.get("out_dtype", torch.bfloat16)
        y = bw_fp8_gemm(B._bw_data, B._bw_scale, A._bw_data, A._bw_scale).to(out_dtype)
        bias = kwargs.get("bias", None)
        if bias is not None:
            y = y + bias.to(out_dtype)
        return y, None, None, None        # matches (out, bias_grad, gelu_in, extra_out)

    # Everything else (dgrad/wgrad, mixed): dequant to bf16 and let TE's gemm do it
    A2 = _dequant_bf16(A) if a_bw else A
    B2 = _dequant_bf16(B) if b_bw else B
    kwargs["quantization_params"] = None
    return _ORIG_GEMM(A2, B2, *args, **kwargs)

tegemm.general_gemm = _general_gemm
telinear.general_gemm = _general_gemm   # linear.py did `from ..cpp_extensions import general_gemm`
# also the modules that import general_gemm by name (QKV via LayerNormLinear, MLP via LayerNormMLP)
import transformer_engine.pytorch.module.layernorm_linear as _teln
import transformer_engine.pytorch.module.layernorm_mlp as _telnmlp
_teln.general_gemm = _general_gemm
_telnmlp.general_gemm = _general_gemm

# ---- 4) fused norm+quantize (QKV/MLP LayerNormLinear): norm in bf16, then our quantize ----
import transformer_engine.pytorch.module._common as _tecommon
_ORIG_APPLY_NORM = _tecommon.apply_normalization
def _apply_normalization(inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer,
                         output_dtype, normalization, fwd_ln_sm_margin, zero_centered_gamma):
    if isinstance(output_quantizer, Float8BlockQuantizer):
        bf16_out = torch.empty(inputmat.shape, dtype=torch.bfloat16, device=inputmat.device)
        ret = _ORIG_APPLY_NORM(inputmat, bf16_out, ln_weight, ln_bias, eps, None,
                               torch.bfloat16, normalization, fwd_ln_sm_margin, zero_centered_gamma)
        dst = ln_out if isinstance(ln_out, Float8BlockwiseQTensorBase) else \
            output_quantizer.make_empty(inputmat.shape, dtype=inputmat.dtype, device=inputmat.device)
        _fill(output_quantizer, ret[0], dst)            # blockwise-quantize the bf16 norm output
        return (dst,) + tuple(ret[1:])
    return _ORIG_APPLY_NORM(inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer,
                            output_dtype, normalization, fwd_ln_sm_margin, zero_centered_gamma)
_tecommon.apply_normalization = _apply_normalization
_teln.apply_normalization = _apply_normalization
_telnmlp.apply_normalization = _apply_normalization

print("[te_blockwise_patch] applied: gate open, quantizer + general_gemm patched (fwd->Triton).")
