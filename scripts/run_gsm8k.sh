#!/bin/bash

# Resolve repo root from this script's location so it can be run from any cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

MODEL_PATH="/data/model/LLaDA-8B-Instruct"
TASK="gsm8k"
BATCH_SIZE=16
MAX_NEW_TOKENS=256
NUM_SHOT=5
NUM_FULL_STEPS=4

# Sweep values
NUM_STEPS_LIST=(256)
THRESHOLD_LIST=(0.99)
BLOCK_SIZE=(32)

for steps in "${NUM_STEPS_LIST[@]}"; do
    tps=$(( MAX_NEW_TOKENS / steps ))
    for thr in "${THRESHOLD_LIST[@]}"; do
        for bs in "${BLOCK_SIZE[@]}"; do
        
            OUTFILE="llada_${TASK}_${steps}_${tps}_${thr}_bs${bs}"
        
            echo "Running: num_steps=$steps threshold=$thr block_size=$bs"
            echo "Output: $OUTFILE"
        
            python -m dyllm.eval.eval \
                --tasks $TASK \
                --batch-size $BATCH_SIZE \
                --model-path $MODEL_PATH \
                --max-new-tokens $MAX_NEW_TOKENS \
                --num-shot $NUM_SHOT \
                --num-steps $steps \
                --num-full-steps $NUM_FULL_STEPS \
                --threshold $thr \
                --output-file $OUTFILE \
                --log-samples \
                --block-size $bs 

            echo "Finished: $OUTFILE"
            echo "--------------------------------------"
        done
    done
done