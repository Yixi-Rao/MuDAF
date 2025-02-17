import os
import torch
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.attn_viz import attention_viz, get_seq_attentions
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
                                attn_implementation = "flash_attention_2",
                                max_memory = {i: f"{max_memory}GiB" for i in range(device_num)})
    model.eval()
    
    return model

# DEFAULT_EOS = [". ", "! ", "? ", "\n"]
DEFAULT_EOS = ["</", "\n"]

def preprocess(text, use_default_eos):
    # 分离出第一句话
    # 排除 Mr. Mrs. Dr. 等缩写的干扰
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
    argparser.add_argument("--check_file", type=str, default="results/longbench_hotpotqa_result_llama3.1_qk_contrast_n.score.json")
    argparser.add_argument("--prompt_file", type=str, default="hotpotqa_e_inputs.jsonl", required=True)
    argparser.add_argument("--dataset", type=str, default="hotpotqa")
    argparser.add_argument("--model", type=str, default="/data/local/EQnA_STCA/Users/v-weihaoliu/models/Llama-3.1-8B-Instruct")
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
    
    # SELECTED_IDS = [0, 6, 14, 19, 25, 27, 28, 38, 41, 159, 175, 183]
    SELECTED_IDS = [10, 15]
    
    for i in SELECTED_IDS:
        item = json_data[i]
        prompt = prompts[i]
        print(f"Start Visulizing... {i} Length: {len(prompt)}")
        # meta_info = item['meta_info']
        # meta_info['chunk_size'] = args.chunk_size
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
            
            chunk_ids.append(p_ids) # ids is the index of next segment.
            passage_index += 1
            
        chunk_ids.append(len(tokenizer.encode(prompt[:prompt.index("\nAnswer the question based on the given passages.")+1])))
        chunk_ids.append(len(tokenizer.encode(prompt)))
        
        print(item)
        print(f"len(tokenized_text): {len(tokenized_text)}")
        print(f"Chunk_ids: {chunk_ids}")
        begin_ids = chunk_ids[-2]
        end_ids = chunk_ids[-1]
        output_length = end_ids - begin_ids
        input_ids = tokenized_text
        
        if device_num == 1:
            input_ids = input_ids.unsqueeze(0).to("cuda")
        
        attentions = get_seq_attentions(model,input_ids, output_length)
        
        for idx in range(output_length+1):
            now_token_id = tokenized_text[begin_ids+idx]
            now_token = tokenizer.decode(now_token_id)
            now_ids = begin_ids + idx
            save_dir = f"visualizations/sequence/{model_name}/{int(i)}/{now_ids}"
            
            if os.path.exists(save_dir):
                continue
            chunk_ids[-1] = now_ids
            now_attentions = attentions[idx]
            meta_info = dict()
            meta_info['output_token'] = True
            meta_info['prompt'] = tokenized_text[:now_ids]
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
                
            attention_viz(now_attentions, meta_info, tokenizer = tokenizer, save_dir=save_dir)
        
        # release memory
        attentions = None