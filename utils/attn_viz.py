import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

def atten_fn(x):
    # e^(x-1)
    EPSILON = 0.05
    ALPHA = 0.5
    # return (np.exp(ALPHA*(x-EPSILON))-np.exp(-ALPHA*EPSILON))/(np.exp(ALPHA*(1-EPSILON))-np.exp(-ALPHA*EPSILON))
    if x < EPSILON:
        return x
    else:
        return np.exp(x-1)

def prefill(model, inputs):
    with torch.no_grad():
        if isinstance(inputs, dict):
            outputs = model(**inputs, output_attentions=False, output_hidden_states=False, use_cache = True, return_dict=True)
        else:
            outputs = model(input_ids=inputs, output_attentions=False, output_hidden_states=False, use_cache = True, return_dict=True)
    return outputs

def decode_one_step(model, q_outputs, inp):
    # flash_attn version < 2.1
    # must visualize step by step.
    """
    :param q_outputs: model outputs. (cached k,v)
    :param inp: next input token
    
    """
    past_kv = q_outputs.past_key_values
    inp = inp.view(1, 1) # inp should be on the same device as model
    with torch.no_grad():
        outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True) # use cache && output attentions of a single token
    next_token_id = outputs.logits[0, -1].argmax()
    return outputs, next_token_id

def get_seq_attentions(model, input_ids, output_length):
    """
    example:
    input_ids = [0,1,2,3,4] 5 tokens
    output_length = 2
    Then begin with [0,1] idx = 1
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    length = input_ids.shape[1]
    
    q_outputs = prefill(model, input_ids[:, :length-output_length-1])
    return_attentions = []
    for idx in range(length-output_length-1, length):
        input_token = input_ids[:, idx]
        q_outputs, next_token_id = decode_one_step(model, q_outputs, input_ids[:, idx])
        return_attentions.append(q_outputs.attentions)
    return return_attentions

def get_last_token_attention(model, input_ids):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    q_outputs = prefill(model, input_ids[:, :-1])
    q_outputs, next_token_id = decode_one_step(model, q_outputs, input_ids[:, -1])
    return q_outputs.attentions

def attention_viz(attentions, meta_info, tokenizer = None, save_dir = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    layer_num = len(attentions)
    chunk_ids = meta_info['question_info'].get('chunk_ids', None)
    if chunk_ids == None:
        print("WARNING: chunk_ids is None")
        
    if 'layers' in meta_info:
        layer_arr = meta_info['layers']
        if layer_arr == 'all':
            layer_arr = list(range(layer_num))
    else:
        layer_arr = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31]
    
    if 'answer_length' in meta_info and meta_info['answer_length'] != 0:
        # TODO: add visualization when generating answers
        print("WARNING: adding answer is not supported now")
    head_num = attentions[0].shape[1]
    chunk_size = meta_info.get('chunk_size', None)
    output_token = meta_info.get('output_token', False)
    prompt = meta_info.get('prompt', None)
    
    if 'length' not in meta_info['question_info']:
        length = attentions[0].shape[2]
    else:
        length = meta_info['question_info']["length"]
        
    with open(f"{save_dir}/question_info.json", "w") as f:
        json.dump(meta_info['question_info'], f, ensure_ascii=False, indent=4)
    
    data = [[] for _ in range(head_num)]
    save_data = [[] for _ in range(head_num)]
    
    for layer in layer_arr:
        if layer >= layer_num:
            print(f"WARNING: layer {layer} is out of range")
            break
        for i, attn_mat in enumerate(attentions[layer][0]):
            # heatmap
            distribution = attn_mat[-1].cpu().detach().numpy()
            attn_list = []
            if chunk_ids is None:
                chunk_num = (distribution.shape[0]-1) // chunk_size+1
                chunk_ids = [i*chunk_size for i in range(chunk_num)]
                if chunk_ids[-1] < distribution.shape[0]-1:
                    chunk_ids.append(distribution.shape[0]-1)
            else:
                chunk_size = "Undefined"
            prepoint = 0
            attend_vector = np.zeros(len(chunk_ids)-2)
            attended_info = []
            for j, point in enumerate(chunk_ids):
                if point <= prepoint:
                    break
                chunk = distribution[prepoint:point]
                attention_value = chunk.sum()
                attn_list.append(attention_value)
                data[i].append({
                    "Layer": layer,
                    "Head": i,
                    "Position": j,
                    "Attention Value": attention_value
                })
                
                save_data[i].append({
                    "Layer": layer,
                    "Head": i,
                    "Position": j,
                    "Attention Value": str(attention_value)
                })
                if j!=0 and j<len(chunk_ids)-1:
                    if attention_value>0.1:
                        attend_vector[j-1] = 1.0
                        indexs = np.argsort(chunk)[::-1]
                        if output_token:
                            sorted_tokens = [tokenizer.decode([prompt[prepoint+idx]]) for idx in indexs]
                            sorted_list = [f"({indexs[idx]}, {sorted_tokens[idx]})  =  {float(chunk[indexs[idx]])}" for idx in range(len(indexs)) if chunk[indexs[idx]]>meta_info['threshold']]
                            
                            attended_item = {
                                "Position": j-1,
                                "Attention Value": float(attention_value),
                                "Sorted Tokens": sorted_list,
                                "Text": tokenizer.decode(prompt[prepoint:point])
                            }
                            attended_info.append(attended_item)
                         
                prepoint = point
                
            if output_token:
                with open(f"{save_dir}/{layer}-{int(i)}_attended_tokens.json", "w") as f:
                    json.dump(attended_info, f, ensure_ascii=False, indent=4)
    
    for i in range(head_num):
        with open(f"{save_dir}/head_{i}.json", "w") as f:
            json.dump(save_data[i], f, ensure_ascii=False, indent=4)
    
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    if head_num == 32:
        num_heads = 32
        nrows = 8
        ncols = 4
    elif head_num == 28:
        num_heads = 28
        nrows = 7
        ncols = 4
    elif head_num == 16:
        num_heads = 16
        nrows = 4
        ncols = 4

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 45))  # Adjust figsize as needed

    for i in range(num_heads):
        # Creating a DataFrame for the current head
        df = pd.DataFrame(data[i])

        # Pivot the DataFrame
        pivot_table = pd.pivot_table(df, values='Attention Value', index=['Layer', 'Position'], aggfunc='mean').reset_index()
        pivot_table = pivot_table.pivot(index="Layer", columns="Position", values="Attention Value")

        # Select the appropriate subplot
        ax = axes[i // ncols, i % ncols]

        # Create the heatmap for the current head
        sns.heatmap(
            pivot_table,
            ax=ax,
            cmap=cmap,
            cbar=False,
            vmin=0,
            vmax=1,
        )

        # Set title and labels for the subplot
        ax.set_title(f'Head {i}', fontsize=10)
        ax.set_xlabel('Position')
        ax.set_ylabel('Layer')

    # Add a big title for the entire figure
    fig.suptitle('Attention Visualization (Context_lenght: {}, Last_Token: {}, Chunk_size: {})'.format(length, tokenizer.decode([prompt[-1]]), chunk_size), fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_file = f'{save_dir}/{length}.png'
    plt.savefig(save_file)
    plt.close()