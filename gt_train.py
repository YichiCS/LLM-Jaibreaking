import os
import torch
import argparse
import numpy as np

from peft import LoraConfig
from transformers import (
    TrainingArguments, 
    Trainer,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
    )

from datasets import load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model




def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--warnings", action="store_true")
    
    parser.add_argument("--exp_name", type=str, default='uc_advbench0050')
    parser.add_argument("--data_root", type=str, default='./data/train')
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--model_save", type=str, default='./checkpoints')
    parser.add_argument("--save_name", type=str, default='uc_advbench0050_all_1epoch')
    
    parser.add_argument("--epochs", type=int, default=1)
    
    
    parser.add_argument("--yichi", action="store_true")
    args = parser.parse_args()
    
    return args


def get_dataset(args):
    
    train_data_path = os.path.join(args.data_root, args.exp_name, 'train.json')
    test_data_path = os.path.join(args.data_root, args.exp_name, 'test.json')
    
    data_files = {
        'train': train_data_path,
        'test': test_data_path,
    }
    
    dataset = load_dataset(path='json', data_files=data_files)
    print(f'[Train Data]: {dataset['train'].num_rows} Samples')
    print(f'[Test Data]: {dataset['test'].num_rows} Samples')
    
    return dataset


def get_model_and_tokenizer(args, device, q_config=None):
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
        use_cache=True,
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=q_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    return model, tokenizer

def main():
    
    args = get_option()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset = get_dataset(args)
    
    def preprocess_function(sample):
        
        template = '<s>[INST] Generate a suffix for the following sentence: {p} \n\nSuffix: {s} [/INST]'
        input = template.format(p=sample['prompt'], s=sample['suffix'])
        return {'text': input}
    
    dataset = dataset.map(
        preprocess_function, 
        remove_columns=dataset["train"].column_names   
    )
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model, tokenizer = get_model_and_tokenizer(args=args, device=device, q_config=quantization_config)
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=64, 
        lora_alpha=16, 
        target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
        lora_dropout=0.1, 
        bias="none", 
        modules_to_save = ["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    model.config.use_cache = False
    
    
    def tokenize(element):
        return tokenizer(
            element["text"],
            truncation=True,
            max_length=256,
            add_special_tokens=False,
        )
        
    dataset_tokenized = dataset.map(
        tokenize, 
        batched=True, 
        num_proc=os.cpu_count(),
        remove_columns=["text"]
    )
    bs=16        # batch size
    ga_steps=1  # gradient acc. steps
    epochs=args.epochs
    steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

    trainargs = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        eval_strategy="steps",
        logging_steps=50,
        eval_steps=steps_per_epoch,
        # save_steps=steps_per_epoch,
        gradient_accumulation_steps=ga_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",
        learning_rate=0.0002,
        group_by_length=True,
        fp16=True,
        ddp_find_unused_parameters=False,
    )
    def collate(elements):
        
        tokenlist=[e["input_ids"] for e in elements]
        tokens_maxlen=max([len(t) for t in tokenlist])  # length of longest input

        input_ids,labels,attention_masks = [],[],[]
        for tokens in tokenlist:
            pad_len=tokens_maxlen-len(tokens)
            
            input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
            labels.append( tokens + [-100]*pad_len )    
            attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

        batch={
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks)
        }
        return batch
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        args=trainargs,
    )

    trainer.train()
    
    peft_model_id=os.path.join(args.model_save, f'Llama2_{args.save_name}')
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    
    print(f'[Model]: {peft_model_id}')

if __name__ == "__main__":
    
    main()