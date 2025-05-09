# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
#     --num_processes=7 src/open_r1/grpo.py \
#     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_base.yaml

#!/bin/bash



# Set custom Hugging Face cache directory, change this to your desired path
export HF_HOME=/modify_into_your_path/hf_cache

export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics

# if you have more gpus, you can apply the distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

# Make sure the cache directory exists
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE

# Launch training
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file SR_R1/accelerate_config/zero3.yaml \
    --num_processes=8 SR_R1/grpo_new.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_base.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
