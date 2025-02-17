import argparse
import json
from tqdm import tqdm
from utils.prompter import LongBenchPromptBuilder, ZeroScrollsPromptBuilder
from utils.eval_util import single_ans_em, has_correct_answer
from datasets import load_dataset
from utils.mp_util import MultiprocessingUtil, ConecatenationMultiprocessingUtil
import multiprocessing
import time
import tiktoken
import logging, os, json
import argparse
from functools import partial

DIR_NAME = os.path.dirname(os.path.abspath(__file__))
DATASET = "hotpotqa"
# DATASET = "2wikimqa"
# DATASET = "qasper"


def load_prompt_dataset(bechmark_namem, dataset_name, data_file):
    with open(os.path.join(data_file), 'r') as f:
        if data_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    if benchmark_name == "longbench":
        prompt_builder = LongBenchPromptBuilder(task_name=dataset_name)
    elif benchmark_name == "zeroscrolls":
        prompt_builder = ZeroScrollsPromptBuilder(task_name=dataset_name)
    else:
        raise ValueError(f"Task name {benchmark_name} is not supported now.")
    
    mp_manager = ConecatenationMultiprocessingUtil(func=prompt_builder.build_prompt,
                                                data=data, n_processes=32)
    processed_data = mp_manager.process_data()
    input_data = processed_data
    return data, input_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=f"longbench_{DATASET}", help="Test data name")
    parser.add_argument("--data_file", type=str, default=f"/home/aiscuser/nfs/projects/dralclm/data/LongBench/{DATASET}.jsonl", help="Test data file")
    parser.add_argument('--save_suffix', type=str, default="default", required=False)
    args = parser.parse_args()
    
    benchmark_name = args.dataset_name.split("_")[0]
    dataset_name = '_'.join(args.dataset_name.split("_")[1:])
    print(f"Benchmark name: {benchmark_name}, Dataset name: {dataset_name}")
    test_data, input_data = load_prompt_dataset(benchmark_name, dataset_name, args.data_file)
    with open(f"{DATASET}_inputs.jsonl", "w") as f:
        for item in input_data:
            f.write(json.dumps(item)+"\n")