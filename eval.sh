NUM_GPUS=1  # You can increase this if you want
MODEL=/mnt/proj2/dd-24-62/self_rewarding/open-r1-main/data/Qwen2.5-1.5B-Open-R1-GRPO-noselfreward-0.25dataset
# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,max_num_batched_tokens=32768,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,max_num_batched_tokens=32768,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"


OUTPUT_DIR=data/evals/diamond/$MODEL

CUDA_VISIBLE_DEVICES=6 lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    # --max-samples 300 \
    --output-dir $OUTPUT_DIR 
    