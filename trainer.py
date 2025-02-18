## Need a code for training with contrastive learning.
from path_tools import *
from typing import List
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import Trainer
import os
DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = True
VERBOSE = False
PAR_DIR = os.path.dirname(DIR)
import torch.distributed as dist
import random
NIL_DATASET = False
from transformers import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from dataset.hotpotqa import HotpotQA, MixedDataset
random.seed(19260817)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = ""
DEFAULT_UNK_TOKEN = "</s>"


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.SiLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@dataclass
class DataCollatorForPQWithNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        bs = len(features)
        queries = [e[0] for e in features]
        passages = [e[1] for e in features]
        input_ids = self.tokenizer(queries, return_tensors=return_tensors, padding=self.padding, truncation=True, max_length=self.max_length,pad_to_multiple_of=self.pad_to_multiple_of)['input_ids']
        labels = self.tokenizer(passages, return_tensors=return_tensors, padding=self.padding, truncation=True, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of)['input_ids']        
        return input_ids, labels


@dataclass
class DataCollatorForContrastiveRetrievalWithNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    max_length : int = 8192
    max_contrast_negative: int = 10
    eos_token: str = DEFAULT_EOS_TOKEN
    
    
    def form_qa_input(self, question, answer, passages, output_answer = True):
        body='\n\n'.join(passages)
        output_str = f'{DEFAULT_BOS_TOKEN} Based on the following passages, answer the question.\n\n<passages>\n{body}\n\n</passages>\n\n'
        output_str += f'Question: {question}\nAnswer:'
        
        if output_answer:
            output_str += f' {answer}{self.eos_token}'
            
        
        return output_str
    
    def tokenized_qa_input(self, question, answer, all_passages):
        qa_input = self.form_qa_input(question, answer, all_passages)
        return self.tokenizer([qa_input], return_tensors=self.return_tensors, padding=self.padding, truncation=True, pad_to_multiple_of=self.pad_to_multiple_of)
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        bs = len(features)
        if bs>1:
            print(bs)
            print(features[1])
            raise ValueError(f"Batch size larger than 1 is not supported now. Got {bs}")
        
        # Contrastive Learning Examples
        # Q <=> anchor
        # P <=> positive
        # in-batch N <=> negative
        # NOTE: Now concatenate all positive passages
        contrast = features[0]['contrast']
        anchor = contrast['anchor']
        contrast_positive = contrast['positive']
        negative = contrast['negative']
        
        if len(contrast_positive) == 0:
            raise ValueError("Positive samples should not be empty.")
        if len(negative) == 0:
            raise ValueError("Negative samples should not be empty.")
        
        if len(negative) > self.max_contrast_negative:
            contrast_negative = random.sample(negative, self.max_contrast_negative)
        else:
            contrast_negative = negative
        
        ### NOTE: For padding_right.
        
        prompt_passages = contrast_positive + contrast_negative
        
        contrast_input_ids = self.tokenizer([anchor], return_tensors=return_tensors, padding=self.padding, 
                                   truncation=True, max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of)['input_ids']
        
        contrast_labels = self.tokenizer(prompt_passages, return_tensors=return_tensors, padding=self.padding, 
                                truncation=True, max_length=self.max_length, 
                                pad_to_multiple_of=self.pad_to_multiple_of)['input_ids']
        
        # P + Q: A:
        qa_data = features[0]['qa']
        question = qa_data['question']
        answer = qa_data['answer']
        qa_positive = qa_data['positive_passages']
        qa_negative = qa_data['negative_passages']
        all_passages = qa_positive + qa_negative
        while self.tokenized_qa_input(question, answer, all_passages)['input_ids'].shape[1] > self.max_length:
            all_passages.pop(-1)
                
        random.shuffle(all_passages)
        
        qa_tokenized = self.tokenized_qa_input(question, answer, all_passages)

        qp_ids = self.tokenizer([self.form_qa_input(question, answer, all_passages, output_answer=False)], return_tensors=return_tensors,
                                truncation=True, max_length=self.max_length)

        end_idx = qp_ids.input_ids.shape[1]
        qa_input_ids = qa_tokenized.input_ids[0]
        qa_attention_mask = qa_tokenized.attention_mask[0]
        qa_labels = torch.where(qa_input_ids != self.tokenizer.pad_token_id, qa_input_ids, -100)
        qa_labels[:end_idx] = -100
        
        if torch.all(qa_labels == -100):
            raise ValueError("All labels are -100.")
        
        return {
            "contrast" : {
                "input_ids": contrast_input_ids,
                "labels": contrast_labels
            },
            "qa": {
                "input_ids": qa_input_ids.unsqueeze(0),
                "attention_mask": qa_attention_mask.unsqueeze(0),
                "labels": qa_labels.unsqueeze(0)
            }
        }

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_file: str = field(default=None, metadata={"help": "train file name"})
    val_file: str = field(default=None, metadata={"help": "val file name"})
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    cutoff_len: int = field(
        default = 32,
        metadata={"help": "cutoff length for training"}
    )
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_neg_sentence: bool = field(
        default=False,
        metadata={"help": "use negative sentence for training"}
    )
    max_contrast_negative: int = field(
        default=10,
        metadata={"help": "maximum number of negative samples for contrastive learning"}
    )
    alpha: float = field(
        default=0.3,
        metadata={"help": "alpha for contrastive loss weight"}
    )
    contrast_heads: str = field(
        # default = None,
        default = None,
        metadata={"help": "heads for contrastive learning. Example: '0-1,2-3'"}
    )
    separate_loss: bool = field(
        default = False,
        metadata={"help": "separate loss for each head"}
    )
    add_proj: bool = field(
        default = False,
        metadata={"help": "add projection layer"}
    )
    eos_token: str = field(
        default = DEFAULT_EOS_TOKEN,
        metadata={"help": "end of sentence token"}
    )            

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = kwargs['model'].config
        self.avg_qa_len = 0

    def compute_loss(self, model, inputs, return_outputs=False):  
        contrast = inputs.pop("contrast")
        qa = inputs.pop("qa")
        queries = contrast["input_ids"]
        passages = contrast["labels"]
        ##################################### For CL #########################################
        
        if self.args.alpha > 0 :
            contrast_heads = self.args.contrast_heads
            heads = len(contrast_heads)
            _, Q_reps = model(queries, return_dict=True, return_Q = True, return_heads = contrast_heads)
            _, K_reps = model(passages, return_dict=True, return_K = True, return_heads = contrast_heads)
            
            if self.is_world_process_zero() and VERBOSE:
                print(f'Q_reps: {Q_reps.shape}, K_reps: {K_reps.shape}, contrast_heads: {contrast_heads}')
            pd_idx = torch.where(queries[0] == self.config.pad_token_id)[0]
            if pd_idx.shape[0] == 0:
                q_idx = 0
            else:
                q_idx = pd_idx[0].item()
            Q_reps = Q_reps[:, q_idx-1, :]
            K_reps = torch.stack([t[torch.where(passages[i]!=self.config.pad_token_id, True, False)].mean(dim=0) for i,t in enumerate(K_reps)]) # mean of non-padding tokens
            
            loss_fn = nn.CrossEntropyLoss()
            if not hasattr(model, "sim"):
                self.sim = Similarity(temp=0.05)
            if self.args.separate_loss:
                Q_reps = Q_reps.view(-1, heads, Q_reps.shape[-1]//heads)
                K_reps = K_reps.view(-1, heads, K_reps.shape[-1]//heads)
                labels = torch.arange(Q_reps.shape[0]).long().to(queries.device)
                loss_contrast = torch.tensor([loss_fn(self.sim(Q_reps[:,i,:].unsqueeze(1), K_reps[:,i,:].unsqueeze(0)),labels) for i in range(heads)]).mean() # mean or sum?
            else:
                sim_mat = self.sim(Q_reps.unsqueeze(1), K_reps.unsqueeze(0))
                labels = torch.arange(sim_mat.size(0)).long().to(queries.device)
                
                loss_contrast = loss_fn(sim_mat, labels)
        else:
            loss_contrast = torch.tensor(0.0).to(queries.device)
        
        #################################### For QA #################################################
        if self.args.alpha < 1:
            try:
                loss_qa = super().compute_loss(model, qa, return_outputs=False)
            except Exception as e:
                print(f"Error: {e}")
                print(f"QA Shape: {qa['input_ids'].shape[1]}")
                raise e
        else:
            loss_qa = torch.tensor(0.0).to(qa["input_ids"].device)
            
        # return loss_qa
        loss_contrast_weighted = self.args.alpha * loss_contrast
        loss_qa_weighted = (1.0 - self.args.alpha) * loss_qa
        loss = loss_contrast_weighted + loss_qa_weighted
        
        if self.is_world_process_zero() and DEBUG:
            self.log({"steps": self.state.global_step, "loss_contrast": loss_contrast_weighted.item(), "loss_qa": loss_qa_weighted.item(), "shape_qa": qa["input_ids"].shape[1]})
            if torch.isnan(loss_contrast_weighted):
                print(contrast)
                print(f'Q_reps: {Q_reps}')
                print(f'shape contrast: {Q_reps.shape}, {K_reps.shape}')
            
            if torch.isnan(loss_qa_weighted):
                print(qa)
                print(f'shape qa: {qa["input_ids"].shape}')
                print(f'shape labels: {qa["labels"].shape}')
            
        
        if self.is_world_process_zero() and VERBOSE:
            if not torch.isnan(loss_contrast_weighted) and not torch.isinf(loss_contrast_weighted):
                self.loss_contrast_accum += loss_contrast_weighted
            else:
                self.loss_contrast_accum +=  self.loss_contrast_accum / (1 + self.state.global_step - self._globalstep_last_logged)
                
                
            if not torch.isnan(loss_qa_weighted) and not torch.isinf(loss_qa_weighted):
                self.loss_qa_accum += loss_qa_weighted
            else:
                self.loss_qa_accum += self.loss_qa_accum / (1 + self.state.global_step - self._globalstep_last_logged)
        
        return loss

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.separate_loss:
        print("Separate Loss is enabled.")
    else:
        print("Separate Loss is disabled.")
        
    if training_args.contrast_heads is not None:
        training_args.contrast_heads = training_args.contrast_heads.split(",")
        
    batch_size = training_args.per_device_train_batch_size
    
    if batch_size > 1:
        print(f"Batch size larger than 1 is not supported now. Got {batch_size}")
        raise ValueError("Batch size larger than 1 is not supported now.")
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=model_args.use_fast_tokenizer,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset = MixedDataset(split="train", file_path=data_args.data_path, file_name=data_args.train_file)
    
    collator = DataCollatorForContrastiveRetrievalWithNeg(tokenizer=tokenizer, 
                                                          max_length=training_args.model_max_length, 
                                                          max_contrast_negative=training_args.max_contrast_negative,
                                                          eos_token=training_args.eos_token)
    
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.pad_token_id = tokenizer.pad_token_id
    print ("Start Building Trainer............")
    trainer = ContrastiveTrainer(model=model, tokenizer=tokenizer, args=training_args, 
                                 data_collator=collator, train_dataset=train_dataset)
    
    model.config.use_cache = False
    print ("Starting Training..........")
    if training_args.resume_from_checkpoint == "null":
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
    trainer.save_state()
    
    
if __name__ == '__main__':
    main()