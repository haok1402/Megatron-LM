#!/bin/bash
# Tokenize one DCLM-baseline shard with the Qwen3.5 tokenizer into Megatron format.

# Launch command for running the tokenizer from the root of this repository:
# srun_mcore --jobid=<id> -N1 -n1 --cpus-per-task=64 -o logs/pretraining/qwen3.5-122b-a10b/setup-%J.log benchmarks/pretraining/qwen3.5-122b-a10b/setup.sh

set -euo pipefail

export HF_HOME=workspace/huggingface
export PYTHONUNBUFFERED=1

SHARD=global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl

hf download mlfoundations/dclm-baseline-1.0 $SHARD.zst --repo-type dataset --local-dir workspace/dclm-baseline
hf download Qwen/Qwen3.5-122B-A10B tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja --local-dir workspace/qwen35-122b-a10b

zstd -d -f workspace/dclm-baseline/$SHARD.zst -o workspace/dclm-baseline/$SHARD
/opt/venv/bin/python tools/preprocess_data.py --input workspace/dclm-baseline/$SHARD --output-prefix workspace/dclm-baseline/qwen35 --json-keys text --tokenizer-type HuggingFaceTokenizer --tokenizer-model workspace/qwen35-122b-a10b --append-eod --workers 64
