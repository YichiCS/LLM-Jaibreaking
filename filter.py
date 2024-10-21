
import os
import random
import argparse
import numpy as np
import pandas as pd

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import PromptDataset
from utils import load_model_and_tokenizer, clean_answer, csv_process, not_matched, seed_everything

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--seed", type=int, default=12)
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    
    parser.add_argument("--root", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    
    parser.add_argument("--yichi", action="store_true")
    
    args = parser.parse_args()
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device]: {device}")
    if args.seed != -1:
        seed_everything(args.seed)
        print(f"[Seed]: {args.seed}")
        
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )
    
    model.eval()
    eval_root = os.path.join(args.root, 'evaluate')
    os.makedirs(eval_root, exist_ok=True)
    
    data = pd.read_csv("./data/advbench/harmful_behaviors_custom.csv")
    goal = data['goal'].tolist()
    
    
    for pid in tqdm(range(args.start, args.end), ncols=100):
        x = goal[pid]
        
        _path =  os.path.join(args.root, f'prompt_{pid}.csv')
        
        try:
            _data = pd.read_csv(_path)
        except:
            print(f'Not exists {_path}')
            continue
        
        _data = csv_process(_data, x)
        data = PromptDataset(_data ,tokenizer)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        
        result = []
        total_success = 0
        columns = ['prompt', 'suffix', 'output', 'ppl', 'result']
        
        print(f'\n\n[Query {pid}]: {x} [Suffix Sum]: {len(data)}')
        save_path = os.path.join(eval_root, f'prompt_{pid}.csv')
        if os.path.exists(save_path):
            print(f'[Result File]: {save_path}')
            continue
        try:
            for prompt, input_ids, attention_mask in tqdm(loader, ncols=100):
                
                prompt = list(prompt)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=0.7, 
                    max_length=512, 
                    pad_token_id=tokenizer.pad_token_id, 
                    do_sample=True, 
                    top_k=10, 
                )
                output_ids = output_ids[:, input_ids.shape[1]:]
                
                for i in range(output_ids.shape[0]):
                        
                    output_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    target_ids = input_ids[i].clone().to(device)
                    ppl_prompt = model(input_ids[i].unsqueeze(0), labels = target_ids).loss
                    ppl_prompt = torch.exp(ppl_prompt).detach().cpu().item()
                
                    output_text_clean = clean_answer(output_text.replace("\n\n", " "))
                    if not_matched(output_text_clean):
                        success = True
                        total_success += 1
                    else:
                        success = False
                        
                    result.append(
                        [x, 
                        prompt[i][len(x):], 
                        output_text_clean.replace("\n", " "), 
                        ppl_prompt, 
                        success]
                    )
            
                import pdb
                pdb.set_trace()
                
            result_data = pd.DataFrame(result, columns=columns)
            result_data.to_csv(save_path)
            
        except Exception as e:
            print(f'[ERROR @ {pid}]: {e}')

if __name__ == "__main__":
        
    main()
    