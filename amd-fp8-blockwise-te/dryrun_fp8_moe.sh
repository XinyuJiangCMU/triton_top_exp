#!/bin/bash
# 5-layer Qwen3-30B-A3B MoE, mock data, random weights, FP8 block-wise dry-run on 1 GPU.
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
cd /root/Megatron-LM

MODEL_ARGS=(
  --disable-bias-linear --qk-layernorm
  --group-query-attention --num-attention-heads 32 --num-query-groups 4 --kv-channels 128
  --num-layers 5 --hidden-size 2048 --ffn-hidden-size 6144
  --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6
  --rotary-percent 1.0 --rotary-base 1000000 --swiglu
  --untie-embeddings-and-output-weights --vocab-size 151936
  # MoE
  --moe-ffn-hidden-size 768 --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall --moe-router-topk 8
  --moe-layer-freq "[1,1,1,1,1]" --num-experts 128 --moe-grouped-gemm
  --moe-router-dtype fp32 --moe-aux-loss-coeff 0
)

FP8_ARGS=(
  --transformer-impl transformer_engine
  --bf16
  --fp8-format e4m3
  --fp8-recipe blockwise
)

TRAIN_ARGS=(
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --micro-batch-size 1 --global-batch-size 1
  --seq-length 256 --max-position-embeddings 4096
  --train-iters 3 --lr 1e-4 --lr-decay-style constant --min-lr 1e-5
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0
  --init-method-std 0.02 --clip-grad 1.0
  --attention-dropout 0.0 --hidden-dropout 0.0
  --tokenizer-type NullTokenizer
  --mock-data
  --no-masked-softmax-fusion --no-gradient-accumulation-fusion
  --no-rope-fusion --no-bias-dropout-fusion
  --eval-iters 0 --eval-interval 1000 --split "100,0,0"
  --log-interval 1 --log-throughput
  --distributed-backend nccl
)

torchrun --nproc-per-node 1 --master-port 29577 \
  /root/pretrain_fp8.py \
  "${MODEL_ARGS[@]}" "${FP8_ARGS[@]}" "${TRAIN_ARGS[@]}"
