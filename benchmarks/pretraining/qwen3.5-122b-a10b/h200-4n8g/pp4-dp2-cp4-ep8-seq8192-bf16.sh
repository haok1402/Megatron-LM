#!/bin/bash
# Benchmark Qwen3.5-122B-A10B pretraining with Megatron-LM.
# 4 nodes x 8 H200, PP=4, DP=2, CP=4, EP=8, seq=8192, bf16.

# Launch command for running the benchmark from the root of this repository:
# srun_mcore --jobid=<id> -N4 --ntasks-per-node=1 -o logs/pretraining/qwen3.5-122b-a10b/h200-4n8g/pp4-dp2-cp4-ep8-seq8192-bf16-%J.log benchmarks/pretraining/qwen3.5-122b-a10b/h200-4n8g/pp4-dp2-cp4-ep8-seq8192-bf16.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1
export TRITON_CACHE_DIR=/tmp/triton-$USER
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_GRAPH_REGISTER=0

declare -A MODEL_CONFIG
MODEL_CONFIG[tokenizer-type]=NullTokenizer
MODEL_CONFIG[vocab-size]=248320
MODEL_CONFIG[hidden-size]=3072
MODEL_CONFIG[ffn-hidden-size]=1024
MODEL_CONFIG[num-attention-heads]=32
MODEL_CONFIG[group-query-attention]=true
MODEL_CONFIG[num-query-groups]=2
MODEL_CONFIG[kv-channels]=256
MODEL_CONFIG[attention-output-gate]=true
MODEL_CONFIG[position-embedding-type]=rope
MODEL_CONFIG[rotary-base]=10000000
MODEL_CONFIG[rotary-percent]=0.25
MODEL_CONFIG[max-position-embeddings]=8192
MODEL_CONFIG[normalization]=RMSNorm
MODEL_CONFIG[norm-epsilon]=1e-6
MODEL_CONFIG[attention-dropout]=0.0
MODEL_CONFIG[hidden-dropout]=0.0
MODEL_CONFIG[disable-bias-linear]=true
MODEL_CONFIG[swiglu]=true
MODEL_CONFIG[untie-embeddings-and-output-weights]=true
MODEL_CONFIG[enable-experimental]=true
MODEL_CONFIG[linear-num-key-heads]=16
MODEL_CONFIG[linear-num-value-heads]=64
MODEL_CONFIG[linear-key-head-dim]=128
MODEL_CONFIG[linear-value-head-dim]=128
MODEL_CONFIG[linear-conv-kernel-dim]=4
MODEL_CONFIG[num-experts]=256
MODEL_CONFIG[moe-router-topk]=8
MODEL_CONFIG[moe-ffn-hidden-size]=1024
MODEL_CONFIG[moe-shared-expert-intermediate-size]=1024
MODEL_CONFIG[moe-router-dtype]=fp32
MODEL_CONFIG[moe-router-score-function]=softmax
MODEL_CONFIG[moe-router-load-balancing-type]=global_aux_loss
MODEL_CONFIG[moe-aux-loss-coeff]=0.001

declare -A INFRA_CONFIG
INFRA_CONFIG[bf16]=true
INFRA_CONFIG[transformer-impl]=transformer_engine
INFRA_CONFIG[spec]="megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec"
BLOCK="GEGEGE*E"; STAGE="$BLOCK|$BLOCK|$BLOCK"
INFRA_CONFIG[hybrid-layer-pattern]="$STAGE|$STAGE|$STAGE|$STAGE"
INFRA_CONFIG[tensor-model-parallel-size]=1
INFRA_CONFIG[pipeline-model-parallel-size]=4
INFRA_CONFIG[context-parallel-size]=4
INFRA_CONFIG[linear-cp-mode]=chunkwise
INFRA_CONFIG[linear-cp-layout]=contiguous
INFRA_CONFIG[expert-model-parallel-size]=8
INFRA_CONFIG[moe-grouped-gemm]=true
INFRA_CONFIG[moe-router-fusion]=true
INFRA_CONFIG[moe-permute-fusion]=true
INFRA_CONFIG[moe-token-dispatcher-type]=alltoall
INFRA_CONFIG[use-distributed-optimizer]=true
INFRA_CONFIG[overlap-param-gather]=true
INFRA_CONFIG[overlap-grad-reduce]=true
INFRA_CONFIG[cross-entropy-loss-fusion]=true
INFRA_CONFIG[cross-entropy-fusion-impl]=native
INFRA_CONFIG[no-create-attention-mask-in-dataloader]=true
INFRA_CONFIG[no-check-for-nan-in-loss-and-grad]=true
INFRA_CONFIG[manual-gc]=true
INFRA_CONFIG[manual-gc-interval]=10
INFRA_CONFIG[cuda-graph-impl]=transformer_engine
INFRA_CONFIG[cuda-graph-modules]="attn moe_router moe_preprocess"
INFRA_CONFIG[te-rng-tracker]=true

declare -A TRAIN_CONFIG
TRAIN_CONFIG[data-path]=workspace/dclm-baseline/qwen35_text_document
TRAIN_CONFIG[split]=1000,0,0
TRAIN_CONFIG[seq-length]=8192
TRAIN_CONFIG[micro-batch-size]=1
TRAIN_CONFIG[global-batch-size]=1024
TRAIN_CONFIG[train-iters]=32
TRAIN_CONFIG[lr]=1e-5
TRAIN_CONFIG[lr-warmup-init]=1e-6
TRAIN_CONFIG[lr-warmup-iters]=32
TRAIN_CONFIG[lr-decay-iters]=33
TRAIN_CONFIG[lr-decay-style]=WSD
TRAIN_CONFIG[lr-wsd-decay-iters]=0
TRAIN_CONFIG[init-method-std]=0.02
TRAIN_CONFIG[optimizer]=adam
TRAIN_CONFIG[eval-iters]=0
TRAIN_CONFIG[eval-interval]=1000
TRAIN_CONFIG[log-interval]=1
TRAIN_CONFIG[log-throughput]=true

LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=$SLURM_NNODES --nproc-per-node=gpu)
LAUNCH_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$(scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1):15213)

MAIN_ARGS=()
for key in ${!MODEL_CONFIG[@]}; do
    val=${MODEL_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done
for key in ${!TRAIN_CONFIG[@]}; do
    val=${TRAIN_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done
for key in ${!INFRA_CONFIG[@]}; do
    val=${INFRA_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done

/opt/venv/bin/python -m torch.distributed.run ${LAUNCH_ARGS[@]} pretrain_hybrid.py ${MAIN_ARGS[@]}
