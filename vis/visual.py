import os
import torch
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.attn_viz import attention_viz, get_last_token_attention
from time import time
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

device_num = torch.cuda.device_count()

def load_hf_model(model_name):
    # get_device memory_size
    print(f"Device num: {device_num}")
    if device_num == 0:
        device = "cpu"
    else:
        device = "cuda"
    max_memory = int(0.9*(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                torch_dtype = torch.float16, 
                                device_map = "auto" ,
                                max_memory = {i: f"{max_memory}GiB" for i in range(device_num)})
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
    argparser.add_argument("--prompt_file", type=str, default="hotpotqa_inputs.jsonl", required=True)
    argparser.add_argument("--dataset", type=str, default="hotpotqa")
    argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    argparser.add_argument("--use_default_eos", action="store_true", default=False)
    argparser.add_argument("--save_suffix", type=str, default=None)
    argparser.add_argument("--chunk_size", type=int, default=64)
    args = argparser.parse_args()
    is_chat_model = True if "chat" in args.model.lower() or "instruct" in args.model.lower() or "-it" in args.model.lower() else False
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = load_hf_model(args.model)
    
    model_name = args.model.split("/")[-1]
    if "checkpoint" in model_name:
        model_name = args.model.split("/")[-2]
    
    with open(args.check_file, 'r') as f:
        json_data = json.load(f)
    
    with open(args.prompt_file, 'r') as f:
        # readlines
        lines = f.readlines()
        prompts = [json.loads(line) for line in lines]
        
    total_nums = len(prompts)
    max_length = 0
    for i in range(total_nums):
        item = json_data[i]
        prompt = prompts[i]
        save_dir = f"visualizations/{args.dataset}/{model_name}/{int(i)}"
        if os.path.exists(save_dir):
            continue
        
        print(f"Start Visulizing... {i}")
        time1 = time()
        question = prompt.split("Question:")[1].split("Answer:")[0].strip()
        inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_text = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask']
        
        if len(tokenized_text) > max_length:
            max_length = len(tokenized_text)
        passage_index = 1
        chunk_ids = []
        
        while f"Passage {passage_index}:" in prompt:
            str_ids = prompt.index(f"Passage {passage_index}:")
            p_ids = len(tokenizer.encode(prompt[:str_ids]))
            
            chunk_ids.append(p_ids)
            passage_index += 1
        chunk_ids.append(len(tokenizer.encode(prompt[:prompt.index("\nAnswer the question based on the given passages.")+1])))
        chunk_ids.append(len(tokenizer.encode(prompt)))
        print(item)
        print(f"len(tokenized_text): {len(tokenized_text)}")
        print(f"Chunk_ids: {chunk_ids}")
        
        
        attentions = get_last_token_attention(model, inputs['input_ids'].to("cuda"))
        
        meta_info = dict()
        meta_info['output_token'] = True
        meta_info['prompt'] = tokenized_text
        meta_info['top-p'] = 0.99
        meta_info['threshold'] = 0.001
        meta_info['question_id'] = i
        meta_info['layers'] = 'all'
        meta_info['question_info'] = {
            "question": question,
            "pred": preprocess(item['pred'], args.use_default_eos),
            "answers": item['answers'],
            "length": item['length'],
            'chunk_ids': chunk_ids,
        }
        
        attention_viz(attentions, meta_info, tokenizer = tokenizer, save_dir=save_dir)
        time2 = time()
        print(f"Time: {time2-time1}s")
        