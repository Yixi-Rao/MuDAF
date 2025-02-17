# generation without vllm

import json
import os
import numpy as np
import torch
import time
import torch
from transformers import AutoModelForCausalLM
device_num = torch.cuda.device_count()
def load_hf_model(model_name, use_flash = True):
    # get_device memory_size
    print(f"Device num: {device_num}")
    if device_num == 0:
        device = "cpu"
    else:
        device = "cuda"
    max_memory = int(0.9*(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))
    
    attn_imp = "flash_attention_2" if use_flash else "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                torch_dtype = torch.float16, 
                                device_map = "auto" ,
                                attn_implementation = attn_imp,
                                max_memory = {i: f"{max_memory}GiB" for i in range(device_num)})
    model.eval()
    
    return model

def prefill(model, inputs):
    with torch.no_grad():
        if isinstance(inputs, dict):
            outputs = model(**inputs, output_attentions=False, output_hidden_states=False, use_cache = True, return_dict=True)
        else:
            outputs = model(input_ids=inputs, output_attentions=False, output_hidden_states=False, use_cache = True, return_dict=True)
    return outputs

def decode_one_step(model, q_outputs, inp, output_attentions = False, block_list = None):
    # flash_attn version < 2.1
    # must visualize step by step.
    """
    :param q_outputs: model outputs. (cached k,v)
    :param inp: next input token
    
    """
    past_kv = q_outputs.past_key_values
    inp = inp.view(1, 1) # inp should be on the same device as model
    with torch.no_grad():
        outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=output_attentions, block_list = block_list) # use cache && output attentions of a single token
    next_token_id = outputs.logits[0, -1].argmax()
    return outputs, next_token_id

def generate(model, input_ids, output_length, tokenizer = None, block_list = None, eos_list = None):
    """
    example:
    input_ids = [0,1,2,3,4] 5 tokens
    output_length = 2
    Then begin with [0,1] idx = 1
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    length = input_ids.shape[1]
    begin_time = time.time()
    q_outputs = prefill(model, input_ids[:, :-1])
    output = []
    fill_time = time.time()
    inp = input_ids[:, -1]
    for step_i in range(output_length):
        inp = inp.view(1, 1)
        q_outputs, next_token_id = decode_one_step(model, q_outputs, inp, block_list = block_list)
        inp = next_token_id
        output.append(inp.item())
        if tokenizer is not None:
            step_token = tokenizer.decode([inp.item()])
            if eos_list and step_token in eos_list:
                break
    
    generate_time = time.time()
    
    return {
        "fill_time": fill_time - begin_time,
        "generate_time": generate_time - fill_time,
        "output": output,
    }