import os
import shutil

import torch

from nltk.corpus import stopwords
from utils import get_options, load_model_and_tokenizer, seed_everything

stop_words = set(stopwords.words('english'))

def main():
    args = get_options()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"[device]: {device}")
    if args.seed != -1:
        seed_everything(args.seed)
        print(f"[seed]: {args.seed}")
        
    model_path_dicts = {"Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
                        "Vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
                        "guanaco-7b": "TheBloke/guanaco-7B-HF",
                        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2", 
                        }
    model_path = model_path_dicts[args.pretrained_model]
    model, tokenizer = load_model_and_tokenizer(model_path, low_cpu_mem_usage=True, use_cache=False, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
   
    try:
        if "uc" in args.mode:
            from UltraCOLD.attack_uc import attack_generation
            attack_generation(model, tokenizer, device, args)
        else:
            raise NotImplementedError(f'NOT supported args.mode {args.mode}')
    
    except Exception as e:
        print(e)
        
    if len(os.listdir(args.result_path)) == 0:
        shutil.rmtree(args.result_path)

if __name__ == "__main__":
    main()
