# AMD 上 FP8 block-wise 训练 — gap 定位 + 自研 Triton kernel + 实测

> 对应会议:解耦(Megatron+TE 与 Miles 分开)→ 随机权重 block-scale FP8 干跑 → 缺的 kernel 自己用 Triton 写。
> 平台:MI355X (gfx950, ROCm 7.0),运行时 TE = ROCm/TransformerEngine `release_v2.8_rocm`。

## TL;DR
1. **解耦成立**:BF16 下 Megatron+TE 训练在 AMD 上端到端跑通(Qwen3-4B RL,grad_norm=0.0602)。"地基"没问题 —— gap 不是 Miles 的问题,是 pre-training 那层的 kernel 问题。
2. **缺口就一个**:DeepSeek 式 **block-scale FP8 GEMM**。`rocm_gemm.cu` 只认 tensorwise + MXFP8,把 `NVTE_BLOCK_SCALING_1D/2D`(1×128 act / 128×128 weight, FP32 scale)直接拒;hipBLASLt 也没有对应 scale mode。**量化侧 kernel 都在,只缺这个 GEMM。**
3. **印证 Henry 的判断**:这正是"等 hipBLAS 支持 = game over"的点。**我们没等 —— 用 Triton 自己写了**,当天验证。
4. **自研 kernel + 训练实测(MI355X)**:
   - dense `te.Linear`(blockwise FP8)训练收敛,贴 bf16;
   - **MoE**(router + grouped experts)训练收敛 27×,用我们自写的**变长分组** block-scale FP8 Triton kernel;
   - 全程**零重编**,靠 python monkeypatch 让 TE 原生模块走我们的 kernel。

## 缺口定位(NV TE 为 ground truth 对 ROCm fork)
- `common/gemm/rocm_gemm.cu` 配矩阵分支:`is_tensor_scaling`(mode0)✓、`is_mxfp_scaling`(mode1, MXFP8)✓、**else → `NVTE_ERROR("unsupported scaling mode")`**。`NVTE_BLOCK_SCALING_1D=2 / 2D=3`(DeepSeek)落进 else。
- descriptor 段只会把 block scaling 配成 MXFP8 的 `VEC32_UE8M0`,**没有** cublasLt 的 `BLK128x128_32F` 等价物。
- 量化侧不缺:`quantize_transpose_{vector,square}_blockwise.cu` + `float8_blockwise_tensor.py` 都在,只是被 ROCm gate(`fp8.py check_fp8_block_scaling_support → False`)拦着。

## 自研 Triton kernel(forward fp8 / backward bf16,即 miles 的 FP8 定义)
- `bw_fp8_gemm.py` — dense block-scale FP8 GEMM。vs fp8 参考 rel_err 1.66e-3(=bf16 输出舍入),vs 真 bf16 量化误差 3.7%。
- `grouped_fp8_gemm.py` — **变长分组** block-scale FP8 GEMM(MoE expert 层),单 launch 覆盖所有 expert,任意 per-expert token 数(M-mask)。非 %128 分组验证通过。

## TE 集成(monkeypatch,免重编)
ROCm TE 在 4 处 C++ 入口拒 block scaling,各 patch 一个:`tex.quantize`(dense 量化)、`tex.split_quantize`(分组量化)、`general_gemm`(dense GEMM,前向走 Triton)、`apply_normalization`(QKV/MLP 的 fused norm+量化,改成 bf16 norm + 我们量化)、`general_grouped_gemm`(MoE GEMM,前向走变长分组 Triton)。
- dense:`te.Linear` + Float8BlockScaling,fwd+bwd+optimizer,loss 1.036→0.743 单调。
- MoE:router + 4 grouped experts,loss 0.509→0.017(27×)。

## 真模型干跑(进行中)
5-layer Qwen3-30B-A3B(128 experts/topk8)+ mock-data + 随机权重 + `--fp8-recipe blockwise`,单卡 Megatron。**已能建模型(3.7B params)、fp8 QKV+norm 路径走通**;当前卡在 Megatron 的 `bias_dropout_add` shape(配置层,非 kernel)。继续调。

## 怎么做(先后)
- **P0 已落地**:block-scale FP8 GEMM(dense + grouped)Triton 实现 + TE 集成 + 单元级训练验证。
- **进行中**:真 Megatron(Qwen3-30B-A3B 5-layer)端到端干跑。
- **后续**:native fp8 tensor-core MMA(性能版,现为 correctness-first);把 backward 也按需 fp8;跟 ROCm/hipBLASLt 并行沟通他们的 block-scale 计划。

## 产物(可复现,容器 hai-te:/root + host /mnt/data/data/hai)
`bw_fp8_gemm.py` · `grouped_fp8_gemm.py` · `te_blockwise_patch.py` · `te_moe_patch.py` ·
`train_fp8_blockwise.py` · `te_native_fp8_test.py` · `moe_fp8_test.py` · `dryrun_fp8_moe.sh`
gap 分析:`te-gap/te_gap_report.md`
