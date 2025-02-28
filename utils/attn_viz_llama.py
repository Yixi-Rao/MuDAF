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
    '''用于将提示传入模型并获取模型的输出 (cached k,v)。'''
    with torch.no_grad():
        if isinstance(inputs, dict):
            # output_attentions=False 和 output_hidden_states=False 表示我们不需要模型的注意力矩阵和隐藏状态，只需要模型的基本输出。
            outputs = model(
                **inputs, 
                output_attentions = False, 
                output_hidden_states = False, 
                use_cache = True, 
                return_dict = True
            )
        else:
            outputs = model(
                input_ids=inputs, 
                output_attentions=False, 
                output_hidden_states=False,
                use_cache = True, 
                return_dict=True
            )
            
    return outputs

def decode_one_step(model, q_outputs, inp):
    # flash_attn version < 2.1
    # must visualize step by step.
    """
    给定下一个 token，解码模型的一步的输出，生成生成的token和attention列表
    
    :param q_outputs: model outputs. (cached k,v)
    :param inp: next input token
    
    """
    # 上一次解码的缓存，帮助模型在后续的步骤中更高效地生成输出
    past_kv = q_outputs.past_key_values
    inp     = inp.view(1, 1)  # inp should be on the same device as model
    
    # use cache then it outputs attentions of a single token （output_attentions）
    with torch.no_grad():
        outputs = model(
            input_ids=inp, 
            past_key_values=past_kv, 
            use_cache=True, 
            output_attentions=True
        ) 
    
    # 表示获取当前输出的最大概率索引，作为下一步生成的token。
    next_token_id = outputs.logits[0, -1].argmax()
    
    return outputs, next_token_id

def get_seq_attentions(model, input_ids, output_length):
    """
    获取序列生成过程中每一步的注意力矩阵
    
    example:
    input_ids = [0,1,2,3,4] 5 tokens
    output_length = 2
    Then begin with [0,1] idx = 1
    """
    # 确保输入的维度为[batch_size, seq_length]，即添加一个批次维度。
    if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
    
    length = input_ids.shape[1]
    
    # q_outputs 存储模型的提示词部分缓存信息，从而进行高效的推理。
    q_outputs = prefill(
        model, 
        input_ids[:, :length - output_length - 1]
    )
    
    # 实现逐步解码并获取每一步的注意力信息。
    return_attentions = []
    for idx in range(length - output_length - 1, length):
        q_outputs, next_token_id = decode_one_step(model, q_outputs, input_ids[:, idx])
        return_attentions.append(q_outputs.attentions)
        
    return return_attentions

def get_last_token_attention(model, input_ids):
    '''用于获取生成序列中最后一个token的注意力矩阵。'''
    if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
    
    # 输入 input_ids[:, :-1] 表示去除最后一个token的输入，q_outputs表示缓存的模型输出。
    q_outputs = prefill(model, input_ids[:, : -1])
    q_outputs, next_token_id = decode_one_step(model, q_outputs, input_ids[:, -1])
    
    return q_outputs.attentions

def attention_viz(attentions, meta_info, tokenizer = None, save_dir = None):
    '''可视化模型在不同层和不同注意力头上的注意力分布
    
        attentions：[num_layers, batch_size, num_heads, seq_length, seq_length]
        meta_info ：包含关于当前输入、问题、输出等的信息，帮助在可视化时正确地理解数据。
        tokenizer：用于将token ID转换回自然语言文本的工具
        save_dir：保存生成的可视化图像的文件夹路径。
    '''
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    layer_num = len(attentions)
    # chunk_ids 是每一个段落的开始和结束位置的索引。
    chunk_ids = meta_info['question_info']['chunk_ids']
    
    # layer_arr 用于选择要可视化的层数
    if 'layers' in meta_info:
        layer_arr = meta_info['layers']
        if layer_arr == 'all': layer_arr = list(range(layer_num))
    else:
        layer_arr = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31]
    
    if 'answer_length' in meta_info and meta_info['answer_length'] != 0:
        # TODO: add visualization when generating answers
        print("WARNING: adding answer is not supported now")
    
    head_num     = attentions[0].shape[1]  # head_num表示注意力头的数量
    chunk_size   = meta_info.get('chunk_size', None)  # chunk_size 表示每个段落的长度
    output_token = meta_info.get('output_token', False)  # output_token 表示是否需要输出token的详细信息
    prompt       = meta_info.get('prompt', None)
    
    if 'length' not in meta_info['question_info']:
        length = attentions[0].shape[2]
    else:
        length = meta_info['question_info']["length"]
        
    with open(f"{save_dir}/question_info.json", "w") as f:
        json.dump(meta_info['question_info'], f, ensure_ascii=False, indent=4)
    
    save_data = [[] for _ in range(head_num)]  # 保存每个头的注意力数据
    data      = [[] for _ in range(head_num)]  # 存储每个注意力头的注意力信息。
    
    #? 获取当前层 layer
    for layer in layer_arr:
        if layer >= layer_num:
            print(f"WARNING: layer {layer} is out of range")
            break
        
        #* 获取当前层 layer 的每个注意力头 i 的注意力矩阵 attn_mat。
        for i, attn_mat in enumerate(attentions[layer][0]):
            # distribution 为 attn_mat[-1] 获取最后一个 token 的注意力列表，转换为NumPy数组进行后续处理
            distribution = attn_mat[-1].cpu().detach().numpy() # heatmap
            
            if chunk_ids is None:
                # 将注意力矩阵按chunk_size进行分块，确保每个块大小不超过chunk_size。
                chunk_num = (distribution.shape[0] - 1) // chunk_size + 1
                chunk_ids = [i * chunk_size for i in range(chunk_num)]
                if chunk_ids[-1] < distribution.shape[0] - 1:
                    chunk_ids.append(distribution.shape[0] - 1)
            else:
                chunk_size = "Undefined"
                
            prepoint      = 0  # 用来追踪当前处理段落开始位置，在每次循环，会更新为当前段落的结束位置，以确保处理下一个块。
            attend_vector = np.zeros(len(chunk_ids) - 2)  # 用来标记段落总注意力值较高的段落
            attended_info = []  # 用来存储每个段落中关注的token信息。这个列表会在后续被填充，尤其是当某个段落的注意力值较高时，会保存该块的详细信息。
            attn_list     = []  # 存储每个段落的注意力值总和。
            
            #! 在每个段落 j 中计算注意力值（块内的所有权重之和）并保存到attn_list中。
            for j, point in enumerate(chunk_ids):
                if point <= prepoint: break
                
                chunk           = distribution[prepoint : point]
                attention_value = chunk.sum()
                
                attn_list.append(attention_value)
                data[i].append({
                    "Layer"          : layer,
                    "Head"           : i,
                    "Position"       : j,
                    "Attention Value": attention_value
                })
                save_data[i].append({
                    "Layer"          : layer,
                    "Head"           : i,
                    "Position"       : j,
                    "Attention Value": str(attention_value)
                })
                
                # j = 0，代表开头指示部分，不需要进行处理。
                if j != 0 and j < len(chunk_ids) - 1:
                    if attention_value > 0.1:
                        attend_vector[j - 1] = 1.0  # 标记当前块的位置为1.0，表示该位置的注意力值较高。
                        indexs = np.argsort(chunk)[::-1]  # 对当前段落的注意力值进行从大到小排序
                        
                        if output_token:
                            # 将 token 从 token ID 转换回原始文本
                            sorted_tokens = [tokenizer.decode([prompt[prepoint + idx]]) for idx in indexs]
                            # 如果注意力值大于阈值，保存排序后的token和它们的注意力值
                            sorted_list   = [f"({indexs[idx]}, {sorted_tokens[idx]})  =  {float(chunk[indexs[idx]])}" 
                                             for idx in range(len(indexs)) if chunk[indexs[idx]] > meta_info['threshold']]
                            
                            attended_item = {
                                "Position"        : j - 1,
                                "Attention Value" : float(attention_value),
                                "Sorted Tokens"   : sorted_list,
                                "Text"            : tokenizer.decode(prompt[prepoint:point])
                            }
                            
                            attended_info.append(attended_item)
                         
                prepoint = point
            
            # 如果需要，保存与token相关的详细信息。    
            if output_token:
                with open(f"{save_dir}/{layer}-{int(i)}_attended_tokens.json", "w") as f:
                    json.dump(attended_info, f, ensure_ascii=False, indent=4)
                    
    # 将每个注意力头的数据保存为JSON文件。
    for i in range(head_num):
        with open(f"{save_dir}/head_{i}.json", "w") as f:
            json.dump(save_data[i], f, ensure_ascii=False, indent=4)
    
    # 创建一个自定义的渐变色图，用于热图的颜色显示。
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # 根据注意力头的数量（head_num），决定将图形分成多少行（nrows）和列（ncols）
    if head_num == 32:
        num_heads = 32
        nrows = 8
        ncols = 4
    elif head_num == 40:
        num_heads = 40
        nrows = 8
        ncols = 5
    elif head_num == 28:
        num_heads = 28
        nrows = 7
        ncols = 4
    elif head_num == 16:
        num_heads = 16
        nrows = 4
        ncols = 4

    # 使用Matplotlib创建一个网格，来绘制多个子图。
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 45))  # Adjust figsize as needed

    # 遍历每个注意力头并生成热图
    for i in range(num_heads):
        # Creating a DataFrame for the current head
        df = pd.DataFrame(data[i])

        # 将 DataFrame 中的数据重新组织。index=['Layer', 'Position']将层和段落位置作为索引，values='Attention Value'表示关心的数值是每个位置的注意力值，aggfunc='mean'表示如果有多个相同位置的值，则求它们的平均值。reset_index()将结果转换为一个标准的DataFrame。
        pivot_table = pd.pivot_table(df, values='Attention Value', index=['Layer', 'Position'], aggfunc='mean').reset_index()
        pivot_table = pivot_table.pivot(index="Layer", columns="Position", values="Attention Value")

        # Select the appropriate subplot
        ax = axes[i // ncols, i % ncols]

        # Create the heatmap for the current head
        sns.heatmap(
            pivot_table,
            ax=ax,
            cmap=cmap,   # 使用之前定义的自定义颜色渐变图cmap来绘制热图。
            cbar=False,  # 不显示颜色条（color bar），因为我们已经知道热图的颜色映射范围。
            vmin=0,      # 设置颜色映射的范围，从0到1，这表示注意力值的强度范围。
            vmax=1,
        )

        # Set title and labels for the subplot
        ax.set_title(f'Head {i}', fontsize=10)
        ax.set_xlabel('Position')
        ax.set_ylabel('Layer')

    # 对每个注意力头的数据，计算注意力值的平均并绘制热图。
    # Add a big title for the entire figure
    fig.suptitle(
        f'Attention Visualization - Context_lenght: {length}, Last_Token: {tokenizer.decode([prompt[-1]])}, Chunk_size: {chunk_size})',
        fontsize=20
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
        
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    save_file = f'{save_dir}/{length}.png'
    
    plt.savefig(save_file)
    plt.close()