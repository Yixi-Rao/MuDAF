import os
import torch
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
from eval.metrics import retrieval_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from utils.attn_viz import attention_viz, get_last_token_attention

device_num = torch.cuda.device_count()

def atten_fn(x):
    # e^(x-1)
    EPSILON = 0.05
    ALPHA = 0.5
    # return (np.exp(ALPHA*(x-EPSILON))-np.exp(-ALPHA*EPSILON))/(np.exp(ALPHA*(1-EPSILON))-np.exp(-ALPHA*EPSILON))
    if x < EPSILON:
        return x
    else:
        return np.exp(x-1)

def get_match(atten, positions):    
    sort_ids = np.argsort(atten)[::-1]
    attend_ids = sort_ids[:len(positions)]
    match_num = sum([1 for p in positions if p in attend_ids])
    if match_num == len(positions) and len(positions) > 0:
        match = 1
    else:
        match = 0
    recall_threshold = 1.0
    for p in positions:
        recall_threshold = min(recall_threshold, atten[p])
        
    precision_threshold = 1.0
    
    for item in sort_ids:
        if item not in positions:
            break
        precision_threshold = atten[item]
    
    return {
        "match": match,
        "recall_threshold": round(float(recall_threshold), 4),
        "precision_threshold": round(float(precision_threshold), 4)
    }
        

def attend_detection(attentions, meta_info, tokenizer = None, save_dir = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    layer_num = len(attentions)
    
    # with open(f"{save_dir}/attention.npy", "wb") as f:
        # np.save(f, attentions)
        
    # import pdb; pdb.set_trace()
    chunk_ids = meta_info['question_info'].get('chunk_ids', None)
    if chunk_ids == None:
        print("WARNING: chunk_ids is None")
        
    if 'layers' in meta_info:
        layer_arr = meta_info['layers']
        if layer_arr == 'all':
            layer_arr = list(range(layer_num))
    else:
        # layer_arr = [16,15,14,13,17,18,32]
        # layer_arr = [1,4,6,8,10,11,12,13,14,15,16,17,18,22,24,26,31,36,41]
        layer_arr = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31]
    
    if 'answer_length' in meta_info and meta_info['answer_length'] != 0:
        # TODO: add visualization when generating answers
        print("WARNING: adding answer is not supported now")
    head_num = attentions[0].shape[1]
    chunk_size = meta_info.get('chunk_size', 1)
    output_token = meta_info.get('output_token', False)
    prompt = meta_info.get('prompt', None)
    
    if 'length' not in meta_info['question_info']:
        length = attentions[0].shape[2]
    else:
        length = meta_info['question_info']["length"]
        
    with open(f"{save_dir}/question_info.json", "w") as f:
        json.dump(meta_info['question_info'], f, ensure_ascii=False, indent=4)
    
    
    golden_positions = meta_info['question_info']['golden_positions']
    pred_positions = meta_info['question_info']['pred_positions']
    # make a 0/1 matrix with shape (layer_num, head_num, len(chunk_ids)-2)
    # attend_matrix = np.zeros((layer_num, head_num, len(chunk_ids)-2))
    data = [[] for _ in range(head_num)]
    save_data = [[] for _ in range(head_num)]
    
    f = open(f"{save_dir}/attend_score.jsonl", "w")
    
    for layer in layer_arr:
        if layer >= layer_num:
            print(f"WARNING: layer {layer} is out of range")
            break
        # print(f"layer {layer}")
        for i, attn_mat in enumerate(attentions[layer][0]):
            # 画矩阵热力图
            distribution = attn_mat[-1].cpu().detach().numpy()
            # distribution is a nd vector
            attn_list = []
            if chunk_ids is None:
                chunk_num = (distribution.shape[0]-1) // chunk_size+1
                chunk_ids = [i*chunk_size for i in range(chunk_num)]
                if chunk_ids[-1] < distribution.shape[0]-1:
                    chunk_ids.append(distribution.shape[0]-1)
            prepoint = 0
            attend_vector = np.zeros(len(chunk_ids)-2)
            for j, point in enumerate(chunk_ids):
                chunk = distribution[prepoint:point]
                attention_value = chunk.sum()
                attn_list.append(attention_value)
                data[i].append({
                    "Layer": layer,
                    "Head": i,
                    "Position": j,
                    "Attention Value": attention_value
                    # "Attention Value": atten_fn(chunk.mean()*chunk_size)
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
                        
                prepoint = point
            
            attend_num = attend_vector.sum()
            # import pdb; pdb.set_trace()
            num = len(chunk_ids)-2
            recall = 0
            for golden in golden_positions:
                if attend_vector[golden] == 1:
                    recall += 1
            
            precision = recall/attend_num if attend_num != 0 else 0
            recall = recall/len(golden_positions)
            f1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
            accuracy = 1 if f1 == 1 else 0
            info_w_golden = get_match(attn_list[1:-1], golden_positions)
            info_w_pred = get_match(attn_list[1:-1], pred_positions)
            dict_obj = {
                "ID": f"{layer}-{int(i)}",
                "Precision": round(precision,4),
                "Recall": round(recall, 4),
                "F1": round(f1,4),
                "Accuracy": round(accuracy,4),
                "Golden Info": info_w_golden,
                "Pred Info": info_w_pred
            }
            f.write(json.dumps(dict_obj, ensure_ascii=False)+"\n")
    
    for i in range(head_num):
        with open(f"{save_dir}/head_{i}.json", "w") as f:
            json.dump(save_data[i], f, ensure_ascii=False, indent=4)
    
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    # Number of heads (heatmaps)
    if head_num == 32:
        num_heads = 32

        # Calculate the grid size (e.g., 4x8 or 8x4)
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
            ax=ax,  # Use the subplot's axes
            cmap=cmap,
            cbar=False,  # Turn off color bar to avoid overlapping (you can add it to one of the subplots if needed)
            vmin=0,
            vmax=1,
        )

        # Set title and labels for the subplot
        ax.set_title(f'Head {i}', fontsize=10)
        ax.set_xlabel('Position')
        ax.set_ylabel('Layer')
        # ax.set_xticks([])  # Optionally remove x-axis ticks
        # ax.set_yticks([])  # Optionally remove y-axis ticks

    # Add a big title for the entire figure
    # fig.suptitle('Attention Visualization on task "Needle In A HayStack" (Chunk_size: {})'.format(chunk_size), fontsize=20)
    fig.suptitle('Attention Visualization (Context_lenght: {}, Chunk_size: {})'.format(length, chunk_size), fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_file = f'{save_dir}/{length}.png'
    plt.savefig(save_file)
    plt.close()


def load_hf_model(model_name):
    # get_device memory_size
    device_num = torch.cuda.device_count()
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

# DEFAULT_EOS = [". ", "! ", "? ", "\n"]
DEFAULT_EOS = ["</", "\n"] # actual </s>

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
    argparser.add_argument("--check_file", type=str, default="detector_results/DirectorQA_result_llama3.1.json")
    argparser.add_argument("--prompt_file", type=str, default="/home/aiscuser/nfs/projects/AEVisualize/RetrievalHeadDetection/directorqa_data.jsonl", required=True)
    argparser.add_argument("--model", type=str, default="/data/local/EQnA_STCA/Users/v-weihaoliu/models/Llama-3.1-8B-Instruct")
    argparser.add_argument("--use_default_eos", action="store_true", default=False)
    argparser.add_argument("--chunk_size", type=int, default=64)
    argparser.add_argument("--start_id", type=int, default=None)
    argparser.add_argument("--golden_nums", type=str, default=None)
    args = argparser.parse_args()
    torch.cuda.empty_cache()
    if args.golden_nums is not None:
        num_list = args.golden_nums.split(',')
        args.golden_nums = [int(num) for num in num_list]
        
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
    
    for i in range(total_nums):
        item = json_data[i]
        if args.start_id is not None and i < args.start_id:
            continue
        
        if (args.golden_nums is not None) and (len(item['Golden Positions']) not in args.golden_nums):
            continue
        
        # save_dir = f"visualizations/{args.check_file.split('_')[0]}/{model_name}/{int(i)}"
        save_dir = f"visualizations/detector32k/{model_name}/{int(i)}"
        if os.path.exists(save_dir):
            continue
        
        data = prompts[i]
        prompt = data['Prompt']
        # prompt = prompts[i]
        meta_info = dict()
        meta_info['chunk_size'] = args.chunk_size
        # if item['meta_info']['depth_percent'] >= 63 or item['meta_info']['real_context_length'] <= 4533:
            # continue
        
        
        # meta_info = item['meta_info']
        # meta_info['chunk_size'] = args.chunk_size
        question = prompt.split("Question:")[1].split("Answer:")[0].strip()
        inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_text = inputs['input_ids']
        print(item)
        print(f"Start Visulizing... {i}, Length: {len(tokenized_text[0])}")
        # with torch.no_grad():
        #     outputs = model(tokenized_text, output_attentions=True, output_hidden_states=False, return_dict=True)
        # attentions = outputs["attentions"]
        
        passage_index = 1
        chunk_ids = []
        pred_positions = []
        pred = preprocess(item['pred'], args.use_default_eos)
        while f"Passage {passage_index}:" in prompt:
            str_ids = prompt.index(f"Passage {passage_index}:")
            p_ids = len(tokenizer.encode(prompt[:str_ids]))
            if prompt[str_ids:].split("\n")[1].strip().lower() in pred.lower():
                pred_positions.append(passage_index-1)
                
            chunk_ids.append(p_ids)
            passage_index += 1
        
        chunk_ids.append(len(tokenizer.encode(prompt[:prompt.index("\n\n\nAnswer the question based on the given passages.")+3])))
        chunk_ids.append(len(tokenizer.encode(prompt)))
        
        print(f"Chunk_ids: {chunk_ids}")
        
        attentions = get_last_token_attention(model, inputs['input_ids'].to("cuda"))
            
        meta_info['output_token'] = True
        meta_info['prompt'] = tokenized_text[0]
        meta_info['top-p'] = 0.99
        meta_info['threshold'] = 0.001
        meta_info['question_id'] = i
        meta_info['layers'] =  'all'
        meta_info['question_info'] = {
            "key": data['Director'],
            "pred": pred,
            "answers": item['answers'],
            "chunk_ids": chunk_ids,
            "golden_positions": item['Golden Positions'],
            "pred_positions": pred_positions,
            "score": retrieval_score(pred, item['answers']),
        }
        
        # if len(tokenizer.encode(prompt))!= len(tokenized_text[0]):
        #     print(f"WARNING: {len(tokenizer.encode(prompt))} != {len(tokenized_text[0])}")
        attend_detection(attentions, meta_info, tokenizer = tokenizer, save_dir=save_dir)
        # import pdb; pdb.set_trace()
