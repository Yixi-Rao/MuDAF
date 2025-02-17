import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

DEFAULT_EOS = ["</s>", "\n","</","\\"]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--pred_dir', type=str, default=".")
    parser.add_argument('--file_names', type=str, default=None, required=True)
    parser.add_argument('--do_preprocess', action='store_true', help="Preprocess the prediction files")
    parser.add_argument('--default_eos', action='store_true', help="Use default EOS symbols")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

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
    args = parse_args()
    scores = dict()
    # if args.e:
    #     path = f"pred_e/{args.model}/"
    # else:
    #     path = f"pred/{args.model}/"
    
    # all_files = os.listdir(path)
    
    file_names = args.file_names.split(";")
    
    print("Evaluating on:", file_names)
    for filename in file_names:
        file_path = os.path.join(args.pred_dir, filename)
        if filename.endswith("jsonl"):
            save_path = os.path.join(args.pred_dir, filename.replace("jsonl", "score.json"))
            with open(file_path, "r", encoding="utf-8") as f:
                 lines = f.readlines()
            data = [json.loads(line) for line in lines]
        else:
            save_path = os.path.join(args.pred_dir, filename.replace("json", "score.json"))
            data = json.load(open(file_path, "r"))
            
        predictions, answers, lengths = [], [], []
        dataset = filename.split('_')[1]
        for item in data:
            if args.do_preprocess:
                predictions.append(preprocess(item["pred"], args.default_eos))
            else:
                predictions.append(item["pred"])
            answers.append(item["answers"])
            all_classes = item["all_classes"]
            if "length" in item:
                lengths.append(item["length"])
        
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
        print(f"{dataset}: {score}")
        with open(save_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        
    # if args.e:
    #     out_path = f"pred_e/{args.model}/result.json"
    # else:
    #     out_path = f"pred/{args.model}/result.json"
    # with open(out_path, "w") as f:
    #     json.dump(scores, f, ensure_ascii=False, indent=4)
