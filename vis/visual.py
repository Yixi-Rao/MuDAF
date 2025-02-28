import os
import torch
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.attn_viz import attention_viz, get_last_token_attention
from time import time

PROMPT = '''
根据所给的多个文档来回答给定的【问题】。只给我【答案】，不要输出任何其他的话，不要继续找问题来自问自答。
以下是给出的【文档】：
{}

【问题】：{}
【答案】：
'''[1:]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

device_num = torch.cuda.device_count()

def load_hf_model(model_name):
    print(f"Device num: {device_num}")
    if device_num == 0:
        device = "cpu"
    else:
        device = "cuda"
        
    max_memory = int(0.9 * (torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype = torch.float16, 
        device_map = "auto" ,
        max_memory = {i: f"{max_memory}GiB" for i in range(device_num)}
    )
    model.eval()
    
    return model

DEFAULT_EOS = ["</", "\n"]

def preprocess(text, use_default_eos):
    if use_default_eos:
        EOS_str = DEFAULT_EOS
    else:
        text = text.replace("Mr. ", "Mr ").replace("Mrs. ", "Mrs ").replace("Dr. ", "Dr ").replace("St. ", "St ").replace("Ms. ", "Ms ").replace("Prof. ", "Prof ").replace("Jr. ", "Jr ").replace("Sr. ", "Sr ").replace("Mr ", "Mr. ").replace("Mrs ", "Mrs. ").replace("Dr ", "Dr. ").replace("St ", "St. ").replace("Ms ", "Ms. ").replace("Prof ", "Prof. ").replace("Jr ", "Jr. ").replace("U.S.", "<US>").strip()
        EOS_str = [". ", "! ", "? ", "\n"]
    
    for symbol in EOS_str:
        if symbol in text:
            text = text.split(symbol)[0]
            
    if not use_default_eos: 
        text = text.replace("Mr ", "Mr. ").replace("Mrs ", "Mrs. ").replace("Dr ", "Dr. ").replace("St ", "St. ").replace("Ms ", "Ms. ").replace("Prof ", "Prof. ").replace("Jr ", "Jr. ").replace("Sr ", "Sr. ").replace("<US>", "U.S.")
        
    return text

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--check_file", type=str, default="results/longbench_hotpotqa_result_Llama3.1-8B-MuDAF.score.json")
    argparser.add_argument("--prompt_file", type=str, required=True)
    argparser.add_argument("--dataset", type=str, default="hotpotqa")
    argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    argparser.add_argument("--use_default_eos", action="store_true", default=False)
    argparser.add_argument("--save_suffix", type=str, default=None)
    argparser.add_argument("--chunk_size", type=int, default=-10)
    args = argparser.parse_args()
    
    tokenizer     = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model         = load_hf_model(args.model)
    
    model_name = args.model.split("/")[-1]
    
    with open(args.prompt_file, 'r') as f:
        lines   = f.readlines()
        prompts = [json.loads(line) for line in lines]
        
    total_nums = len(prompts)
    max_length = 0
    for i in range(total_nums):
        sample   = prompts[i]
        save_dir = f"visualizations/{args.dataset}/{model_name}/{int(i)}"
        if os.path.exists(save_dir):
            continue
        
        print(f"Start Visulizing... {i}")
        
        time1 = time()
        
        question       = sample["question"]
        answer         = sample["answer"]
        prompt         = sample["prompt"]
        
        inputs         = tokenizer(prompt, return_tensors="pt")
        tokenized_text = inputs['input_ids'][0]
        
        max_length = max(len(tokenized_text), max_length)
            
        passage_index = 1
        chunk_ids     = []
        
        while f"文档 [{passage_index}]" in prompt:
            str_ids = prompt.index(f"文档 [{passage_index}] \n")
            p_ids   = len(tokenizer.encode(prompt[:str_ids]))
            
            chunk_ids.append(p_ids)
            passage_index += 1
            
        chunk_ids.append(len(tokenizer.encode(prompt[:prompt.index(f'\n【问题】：{question}') + 1])))
        chunk_ids.append(len(tokenizer.encode(prompt)))

        print(f"len(tokenized_text): {len(tokenized_text)}")
        print(f"Chunk_ids: {chunk_ids}")
        
        attentions = get_last_token_attention(model, inputs['input_ids'].to("cuda"))
        
        meta_info = {
            "output_token" : True,
            "prompt"       : tokenized_text,
            "top-p"        : 0.99,
            "threshold"    : 0.001,
            "question_id"  : i,
            "layers"       : 'all',
            "question_info": {
                "question" : question,
                "answers"  : answer,
                'chunk_ids': chunk_ids,
            }
        }
        
        attention_viz(attentions, meta_info, tokenizer = tokenizer, save_dir=save_dir)
        
        time2 = time()
        
        print(f"Time: {time2-time1}s")
        