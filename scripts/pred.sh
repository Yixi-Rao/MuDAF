export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export VLLM_TRACE_FUNCTION=1
export NCCL_DEBUG=TRACE
export CUDA_VISIBLE_DEVICES=0,1
SAVE_SUFFIX=Llama3.1-8B-MuDAF
MODEL_PATH=models/Llama3.1-8B-MuDAF
TP=2

echo "Evaluating... LongBench"
PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_hotpotqa" \
  --data_file "data/LongBench/hotpotqa.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_2wikimqa" \
  --data_file "data/LongBench/2wikimqa.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_qasper" \
  --data_file "data/LongBench/qasper.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_musique" \
  --data_file "data/LongBench/musique.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_hotpotqa" \
  --data_file "data/LongBench/hotpotqa_e.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX"_e" \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_2wikimqa" \
  --data_file "data/LongBench/2wikimqa_e.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX"_e" \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "longbench_qasper" \
  --data_file "data/LongBench/qasper_e.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX"_e" \
  --model_path $MODEL_PATH

echo "Evaluating... ZeroScrolls"

PYTHONPATH=. python eval/pred.py \
  --dataset_name "zeroscrolls_musique" \
  --data_file "data/ZeroScrolls/musique/test.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH

PYTHONPATH=. python eval/pred.py \
  --dataset_name "zeroscrolls_qasper" \
  --data_file "data/ZeroScrolls/qasper/test.jsonl" \
  --tensor_parallel_size $TP \
  --max_seq_len 30000 \
  --save_suffix $SAVE_SUFFIX \
  --model_path $MODEL_PATH