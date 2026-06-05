"""
MoE extension of te_blockwise_patch: make te.pytorch.GroupedLinear (the expert layer)
run Float8BlockScaling on ROCm. Patches the two grouped C++ entry points:
  - tex.split_quantize        (grouped blockwise quantize)  -> per-split torch quant
  - general_grouped_gemm      (grouped block-scale GEMM)     -> dequant each group to bf16,
                                                                delegate to TE's working
                                                                bf16 grouped gemm (cutlass)
Import AFTER/with te_blockwise_patch.
"""
import torch
import te_blockwise_patch as base                     # applies dense patches + gate
from te_blockwise_patch import _fill, _dequant_bf16
from bw_fp8_gemm import BLK
from grouped_fp8_gemm import grouped_bw_fp8_gemm
import transformer_engine_torch as tex
import transformer_engine.pytorch.cpp_extensions.gemm as tegemm
import transformer_engine.pytorch.module.grouped_linear as tegl
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer, Float8BlockwiseQTensorBase,
)

# ---- grouped quantize: split input by m_splits, quantize each chunk blockwise ----
_ORIG_SPLIT_Q = tex.split_quantize
def _split_quantize(inp, m_splits, quantizers):
    if quantizers and isinstance(quantizers[0], Float8BlockQuantizer):
        outs, off = [], 0
        for i, m in enumerate(m_splits):
            chunk = inp[off:off + m]; off += m
            q = quantizers[i]
            dst = q.make_empty(chunk.shape, dtype=chunk.dtype, device=chunk.device)
            outs.append(_fill(q, chunk, dst))
        return outs
    return _ORIG_SPLIT_Q(inp, m_splits, quantizers)
tex.split_quantize = _split_quantize

# ---- grouped gemm: dequant blockwise operands per expert -> bf16, delegate ----
_ORIG_GGEMM = tegemm.general_grouped_gemm
def _is_bw(t):
    return isinstance(t, Float8BlockwiseQTensorBase) and hasattr(t, "_bw_data")
def _deq_list(L):
    return [(_dequant_bf16(t) if _is_bw(t) else t) for t in L]
def _general_grouped_gemm(A, B, out, *args, **kwargs):
    layout = kwargs.get("layout", "TN")
    single = kwargs.get("single_output", False)
    m_splits = kwargs.get("m_splits", None)
    # FORWARD: weights(dim2) x input chunks(dim1), TN, single concat output -> our Triton kernel
    if (single and layout == "TN" and m_splits is not None
            and len(A) and all(_is_bw(t) and t._bw_dim == 2 for t in A)
            and all(_is_bw(t) and t._bw_dim == 1 for t in B)):
        N, K = A[0]._bw_data.shape
        if N % BLK == 0 and K % BLK == 0 and all(m % BLK == 0 for m in m_splits):
            a_q = torch.cat([b._bw_data for b in B], 0)
            a_sc = torch.cat([b._bw_scale for b in B], 0)
            w_q = torch.stack([a._bw_data for a in A], 0)
            w_sc = torch.stack([a._bw_scale for a in A], 0)
            C = grouped_bw_fp8_gemm(a_q, a_sc, w_q, w_sc, list(m_splits), N)
            out[0].copy_(C.to(out[0].dtype))
            return out[0], None, None
    # dgrad/wgrad/mixed: dequant -> bf16, delegate to TE's grouped gemm
    if any(_is_bw(t) for t in list(A) + list(B)):
        return _ORIG_GGEMM(_deq_list(A), _deq_list(B), out, *args, **kwargs)
    return _ORIG_GGEMM(A, B, out, *args, **kwargs)
tegemm.general_grouped_gemm = _general_grouped_gemm
tegl.general_grouped_gemm = _general_grouped_gemm   # grouped_linear imported it by name

print("[te_moe_patch] applied: split_quantize + general_grouped_gemm patched for blockwise.")
