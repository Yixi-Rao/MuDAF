import sys
import argparse
import json
from tqdm import tqdm
import torch
from utils.prompter import LongBenchPromptBuilder, ZeroScrollsPromptBuilder
from utils.eval_util import single_ans_em, has_correct_answer
from utils.mp_util import MultiprocessingUtil, ConecatenationMultiprocessingUtil
from datasets import load_dataset
import multiprocessing
import time
import tiktoken
import logging, os, json
import argparse
from functools import partial
import vllm
from vllm import LLM, SamplingParams
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
    # mp_manager = ConecatenationMultiprocessingUtil(func=partial(prompt_builder.build_prompt, fit_train=True),
    #                                             data=data, n_processes=32)
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
    parser.add_argument('--tensor_parallel_size', type=int, default=8, required=False)
    parser.add_argument('--trust_remote_code', type=bool, default=True, required=False)
    parser.add_argument('--max_seq_len', type=int, default=2**15, required=False)
    parser.add_argument('--batch_size', type=int, default=2, required=False)
    parser.add_argument('--save_suffix', type=str, default="default", required=False)
    args = parser.parse_args()
    
    benchmark_name = args.dataset_name.split("_")[0]
    dataset_name = '_'.join(args.dataset_name.split("_")[1:])
    print(f"Benchmark name: {benchmark_name}, Dataset name: {dataset_name}")
    test_data, input_data = load_prompt_dataset(benchmark_name, dataset_name, args.data_file)
    
    eval_config = json.load(open(os.path.join(DIR_NAME, "eval_config.json"), "r"))
    if dataset_name not in eval_config:
        eval_config[dataset_name] = {"max_length": 32}
    print(f"Version of VLLM: vllm {vllm.__version__}")
    torch.cuda.empty_cache()
    
    kwargs = dict()
    version_vllm = vllm.__version__
    mid_v = int(version_vllm.split('.')[1])
    if mid_v>=5:
        kwargs['max_seq_len_to_capture'] = args.max_seq_len
    kwargs['max_model_len'] = args.max_seq_len
    llm = LLM(model=args.model_path,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=args.trust_remote_code,
              dtype=torch.float16,
              max_num_batched_tokens=80000,
              gpu_memory_utilization=0.9,
              disable_custom_all_reduce=True,
              **kwargs)
    
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=eval_config[dataset_name]["max_length"])

    batch_size = args.batch_size
    data_number = len(input_data)
    total_batch_num = (len(input_data) // batch_size) + 1

    current_lines = 0
    all_outputs = []

    for batch_idx in tqdm(range(total_batch_num)):
        if batch_idx == total_batch_num-1:
            prompt_batch = input_data[batch_idx * batch_size:]
            test_data_batch = test_data[batch_idx * batch_size:]
        else:
            prompt_batch = input_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
            test_data_batch = test_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
        # print(f"Prompt batch: {prompt_batch}")
        results = llm.generate(prompt_batch, sampling_params)
        current_lines += batch_size
        # print(f"{current_lines} in {data_number} examples.")
        for data, result in zip(test_data_batch, results):
            all_outputs.append({
                'id': data["id"] if "id" in data else None,
                'pred': result.outputs[0].text,
                "answers": data["answers"] if "answers" in data else [],
                "all_classes": data["all_classes"] if "all_classes" in data else None,
                "length": data["length"] if "length" in data else None,
                })
            
    if not os.path.exists(f"{benchmark_name}_results"):
        os.makedirs(f"{benchmark_name}_results")
    # longbench_results
    with open(f"{benchmark_name}_results/{args.dataset_name}_result_{args.save_suffix}.json", "w") as f:
        json.dump(all_outputs, f, indent=4)