SAVE_DIR=models/Llama3.1-8B-MuDAF
SELECTED_HEADS=13-1,13-18,16-1,16-9,16-26,17-21,17-24,17-26
while true; do
    target_folder=$(ls -d "$SAVE_DIR"/checkpoint-* 2>/dev/null | sort | tail -n 2 | head -n 1)
    if [ -n "$target_folder" ]; then
        checkpoint_number=$(basename "$target_folder" | sed 's/^checkpoint-//')
        echo "resumed from checkpoint-$checkpoint_number"
        if ((checkpoint_number > 11200)); then
            echo "finished, exit."
            break
        fi
    else
        echo "cant not find checkpoints to resume from."
        target_folder=null
    fi

    DS_SKIP_CUDA_CHECK=1 deepspeed --num_gpus 8 \
        --num_nodes 2 \
        --master_port 9500 \
        --hostfile configs/hostfile \
        trainer.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_file "hotpot_qa_train_new.jsonl" \
        --data_path "data/hotpotqa" \
        --output_dir $SAVE_DIR \
        --num_train_epochs 2 \
        --model_max_length 8192 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 3 \
        --learning_rate 5e-6 \
        --warmup_ratio 0.05 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --report_to "tensorboard" \
        --gradient_checkpointing True \
        --deepspeed configs/stage2.json \
        --remove_unused_columns False \
        --attn_implementation "flash_attention_2" \
        --bf16 \
        --torch_dtype "bfloat16" \
        --low_cpu_mem_usage 1 \
        --contrast_heads $SELECTED_HEADS \
        --eos_token $EOS_TOKEN \
        --alpha  0.7 \
        --resume_from_checkpoint $target_folder

    sleep 5
done