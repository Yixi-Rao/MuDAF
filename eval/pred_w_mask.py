import sys
import argparse
import json
from tqdm import tqdm
import torch
from utils.prompter import LongBenchPromptBuilder, ZeroScrollsPromptBuilder
from utils.mp_util import MultiprocessingUtil, ConecatenationMultiprocessingUtil
from utils.gen_utils import generate, load_hf_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import multiprocessing
import time
import tiktoken
import logging, os, json
import argparse
from functools import partial

DIR_NAME = os.path.dirname(os.path.abspath(__file__))


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
    parser.add_argument("--dataset_name", type=str, default=None, help="Test data name")
    parser.add_argument("--data_file", type=str, default=None, help="Test data file")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_suffix', type=str, default="default", required=False)
    parser.add_argument('--block_list', type=str, default=None)
    args = parser.parse_args()
    if args.block_list is not None:
        items = args.block_list.split(",")
        block_list = []
        for item in items:
            layer = int(item.split('-')[0])
            head = int(item.split('-')[1])
            block_list.append((layer, head))
        args.block_list = block_list
    benchmark_name = args.dataset_name.split("_")[0]
    dataset_name = '_'.join(args.dataset_name.split("_")[1:])
    print(f"Benchmark name: {benchmark_name}, Dataset name: {dataset_name}")
    test_data, input_data = load_prompt_dataset(benchmark_name, dataset_name, args.data_file)
    eval_config = json.load(open(os.path.join(DIR_NAME, "eval_config.json"), "r"))
    if dataset_name not in eval_config:
        eval_config[dataset_name] = {"max_length": 32}
    max_tokens = eval_config[dataset_name]["max_length"]
    model = load_hf_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    current_lines = 0
    all_outputs = []
    for i in range(len(test_data)):
        prompt = input_data[i]
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
        outputs = generate(model, input_ids, max_tokens, tokenizer, block_list=args.block_list, eos_list=["\n", "\n\n", "\n\n\n"])
        generated_content = tokenizer.decode(outputs['output'], skip_special_tokens=True)
        print(f"Prompt Length: {input_ids.shape[1]}\nOutput: {generated_content.strip()}\nPrefill Time: {outputs['fill_time']} s\nGenerate Time: {outputs['generate_time']} s")
        all_outputs.append({
            'id': test_data[i]["id"] if "id" in test_data[i] else None,
            'pred': generated_content,
            "answers": test_data[i]["answers"] if "answers" in test_data[i] else [],
            "all_classes": test_data[i]["all_classes"] if "all_classes" in test_data[i] else None,
            "length": test_data[i]["length"] if "length" in test_data[i] else None,
        })
            
    if not os.path.exists(f"{benchmark_name}_results"):
        os.makedirs(f"{benchmark_name}_results")
    
    with open(f"{benchmark_name}_results/w_mask/{args.dataset_name}_result_{args.save_suffix}.json", 'w') as f:
        json.dump(all_outputs, f, indent=4)