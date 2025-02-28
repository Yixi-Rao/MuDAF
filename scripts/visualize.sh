#!/bin/bash
# visualize.sh
# 该脚本依次运行 visual.py 与 retrieval_score.py 完成整个 pipeline。

# 设置参数（可根据实际情况修改）
DATASET="CNqa"
# 注意：visual.py 中默认模型参数为 "meta-llama/Llama-3.1-8B"
MODEL=""
# prompt 文件路径，要求每行是一个 JSON 格式的输入（例如 hotpotqa_inputs.jsonl）
PROMPT_FILE="CNqa_inputs.jsonl"

echo "============================================"
echo "Step 1: Running Attention Visualization..."
echo "Dataset: $DATASET, Model: $MODEL, Prompt file: $PROMPT_FILE"
echo "============================================"
# 调用 visual.py，传入必要参数
python visual.py --prompt_file "$PROMPT_FILE" --dataset "$DATASET" --model "$MODEL"

echo "============================================"
echo "Step 2: Running Retrieval Score Evaluation..."
echo "============================================"
# 调用 retrieval_score.py（该脚本内部使用常量配置，如 NUM_LAYERS、NUM_HEADS）
python retrieval_score.py

echo "============================================"
echo "Pipeline Finished."
