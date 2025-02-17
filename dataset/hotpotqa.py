# Code adapted from llm2vec repo
# Ref: https://github.com/McGill-NLP/llm2vec

import json
import random
import os
from torch.utils.data import Dataset

class HotpotQA(Dataset):
    def __init__(
        self,
        dataset_name: str = "HoppotQA",
        split: str = "train",
        file_path: str = "/home/aiscuser/nfs/LongContext/data/HotpotQA",
        file_name: str = "hotpot_qa_train_new.jsonl",
        shuffle_individual_datasets: bool = True,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.data = []
        self.path = os.path.join(file_path, file_name)
        self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        # logger.info(f"Loading {self.dataset_name} data from {self.path}...")
        with open(self.path, "r") as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines if len(json.loads(line)['negative'])>0]
        len_of_data = len(data)
        num_eval = int(len_of_data * 0.1)+1
        if num_eval>100:
            num_eval = 100
        self.eval_data = data[:num_eval]
        self.train_data = data[num_eval:]
        if self.split == "train":
            self.data = self.train_data
        else:
            self.data = self.eval_data
        
        
    def __getitem__(self, index):
        # query, passage
        return (self.data[index]["question"], self.data[index]["answer"], self.data[index]["positive"], self.data[index]["negative"])
    


class MixedDataset(HotpotQA):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.contrast_data = []
        # self.qa_data = self.data
        # repeat 2 times
        # self.qa_data = self.qa_data + self.qa_data
        self.qa_data = []
        for item in self.data:
            anchor = "Question: " + item["question"] + "\nAnswer:"
            self.contrast_data.append({
                    "anchor": anchor,
                    "positive": self.form_passages([item["positive"][0]]),
                    "negative": self.form_passages(item["negative"])
                })

            self.contrast_data.append({
                    "anchor": anchor,
                    "positive": self.form_passages([item["positive"][1]]),
                    "negative": self.form_passages(item["negative"])
                })
            
            self.qa_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "positive_passages": self.form_passages(item["positive"]),
                "negative_passages": self.form_passages(item["negative"])
            })
        
        self.qa_data = self.qa_data + self.qa_data
    
    def form_passages(self, passages):
        formalized_passages = []
        template = "Passage:\n{title}\n{content}"
        for passage in passages:
            fom_pas = template.format(title=passage["title"], content=passage["content"])
            formalized_passages.append(fom_pas)
        return formalized_passages

    def __len__(self):
        return len(self.contrast_data)
    
    def __getitem__(self, index):
        return {
            "contrast": self.contrast_data[index],
            "qa": self.qa_data[index]
        }
