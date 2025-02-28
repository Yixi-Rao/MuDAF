import json
import os
from tqdm import tqdm

MODEL_NAME = 'Qwen-2.5-32B'
DATASET    = "CN_doc"

golden_lable_file = f"{DATASET}_golden_passages_ids.json"
save_dir          = f"visualizations/{DATASET}/{MODEL_NAME}"

with open(golden_lable_file, 'r') as f:
    golden_passages_ids = json.load(f)

f1_scores        = dict()
recall_scores    = dict()
precision_scores = dict()
match_scores     = dict()
num_items        = 0

for layer in range(32):
    for head in range(32):
        f1_scores[f"{layer}-{head}"] = []
        recall_scores[f"{layer}-{head}"] = []
        precision_scores[f"{layer}-{head}"] = []
        match_scores[f"{layer}-{head}"] = []
        
NEW_THRESHOLD = 0.3

for index in tqdm(range(200)):
    golden_id1 = golden_passages_ids[index][0]
    golden_id2 = golden_passages_ids[index][1]
    comment    = golden_passages_ids[index][2]
    
    if os.path.exists(f"{save_dir}/{index}"):
        
        # with open(f"{dir}/{index}/question_info.json", "r") as f:
            # data = json.load(f)
            
        strr = comment.split(';')[0]
        if ',' in strr:
            golden_ids = [int(i) for i in strr.split(',')]
        else:
            golden_ids = []
            if golden_id1 > 0:
                golden_ids.append(golden_id1)
            if golden_id2 > 0:
                golden_ids.append(golden_id2)
        
        for layer in range(32):
            for head in range(32):
                if os.path.exists(f"{save_dir}/{index}/{layer}-{head}/attended_tokens.json"):
                    with open(f"{save_dir}/{index}/{layer}-{head}/attended_tokens.json", "r") as f:
                        data = json.load(f)
                elif os.path.exists(f"{save_dir}/{index}/{layer}-{head}_attended_tokens.json"):
                    with open(f"{save_dir}/{index}/{layer}-{head}_attended_tokens.json", "r") as f:
                        data = json.load(f)
                else:
                    print(f"{save_dir}/{index}/{layer}-{head}/attended_tokens.json not found")
                    continue                    
                
                sorted_data = sorted(data, key=lambda x: x['Attention Value'], reverse=True)
                
                if NEW_THRESHOLD > 0:
                    sorted_data = [item for item in sorted_data 
                                   if item['Attention Value'] > NEW_THRESHOLD]
                    
                attended_ids = [item['Position'] + 1 for item in sorted_data]
                
                sorted_prefix_items = [item['Position']+1 for item in sorted_data[:len(golden_ids)]]
                
                recall = len(set(golden_ids) & set(attended_ids)) / len(golden_ids)
                
                if len(attended_ids) == 0:
                    precision = 0
                else:
                    precision = len(set(golden_ids) & set(attended_ids)) / len(attended_ids)
                if recall + precision == 0:
                    f1 = 0
                else:
                    f1 = 2 * recall * precision / (recall + precision)
                    
                exact_match = 1 if set(golden_ids) == set(sorted_prefix_items) else 0
                
                precision_scores[f"{layer}-{head}"].append(precision)
                recall_scores[f"{layer}-{head}"].append(recall)
                f1_scores[f"{layer}-{head}"].append(f1)
                match_scores[f"{layer}-{head}"].append(exact_match) 

for layer in range(32):
    for head in range(32):
        precision_scores[f"{layer}-{head}"] = sum(precision_scores[f"{layer}-{head}"]) / len(precision_scores[f"{layer}-{head}"])
        recall_scores[f"{layer}-{head}"] = sum(recall_scores[f"{layer}-{head}"]) / len(recall_scores[f"{layer}-{head}"])
        f1_scores[f"{layer}-{head}"] = sum(f1_scores[f"{layer}-{head}"]) / len(f1_scores[f"{layer}-{head}"])
        match_scores[f"{layer}-{head}"] = sum(match_scores[f"{layer}-{head}"]) / len(match_scores[f"{layer}-{head}"])

with open(f"{save_dir}/scores.json", "w") as f:
    json.dump({"precision": precision_scores, "recall": recall_scores, "f1": f1_scores, "match": match_scores}, f, indent=4)
    
sorted_by_exact_match = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
with open(f"{save_dir}/sorted_by_exact_match.json", "w") as f:
    json.dump(sorted_by_exact_match, f, indent=4)

sorted_by_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
with open(f"{save_dir}/sorted_by_f1.json", "w") as f:
    json.dump(sorted_by_f1, f, indent=4)

sorted_by_precision = sorted(precision_scores.items(), key=lambda x: x[1], reverse=True)
with open(f"{save_dir}/sorted_by_precision.json", "w") as f:
    json.dump(sorted_by_precision, f, indent=4)

sorted_by_recall = sorted(recall_scores.items(), key=lambda x: x[1], reverse=True)
with open(f"{save_dir}/sorted_by_recall.json", "w") as f:
    json.dump(sorted_by_recall, f, indent=4)
