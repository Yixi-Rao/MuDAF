FILE_SUFFIX=Llama3.1-8B-MuDAF

PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_hotpotqa_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos

PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_2wikimqa_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos
    
PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_qasper_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos

PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_musique_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos \
   
FILE_SUFFIX=$FILE_SUFFIX"_e"

PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_hotpotqa_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos \
    --e

PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_2wikimqa_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos \
    --e
    
PYTHONPATH=. python eval/eval.py \
    --pred_dir ./longbench_results/ \
    --file_names longbench_qasper_result_$FILE_SUFFIX.json \
    --do_preprocess \
    --default_eos \
    --e \