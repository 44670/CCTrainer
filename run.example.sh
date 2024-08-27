#!/bin/sh

# usage: run.sh 0

export OMP_NUM_THREADS=8
export NCCL_IB_GID_INDEX=3
export WANDB_PROJECT="sft"
export BASE_MODEL="l70i-a"
export CURRENT_NAME="exp1"
export WANDB_NAME=$BASE_MODEL-$CURRENT_NAME
echo "- Node rank: $1"



torchrun --nproc_per_node=8 --nnodes 1 --master_addr "127.0.0.1" --master_port=12346 --node-rank $1 \
    ./sft.py --model_name_or_path /data/output/$BASE_MODEL/final \
             --output_dir /data/output/$WANDB_NAME \
             --tokenizer_path /data/tokenizers/llama3-8b-v2 \
             --train_dataset /data/datasets/exp1.jsonl \
             --eval_dataset  /data/datasets/eval.jsonl  \
             --eval_steps 30 \
             --num_train_epochs 1 \
             --max_length 2500 \
             --per_device_train_batch_size 11 \
             --per_device_eval_batch_size 3 \
             --gradient_accumulation_steps 3 \
             --learning_rate 1e-5 \
            --weight_decay 0.0 \
             --sample_format fourfourml 
