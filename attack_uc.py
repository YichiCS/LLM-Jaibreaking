
import os
import pandas as pd

from nltk.corpus import stopwords

from utils import _get_keywords
from utils import get_sys_prompt
from decode_uc import decode

stop_words = set(stopwords.words('english'))

def attack_generation(model, tokenizer, device, args, model_back=None):
    
    data = pd.read_csv("./data/advbench/harmful_behaviors_custom.csv")
    targets = data['target'].tolist()
    goals = data['goal'].tolist()
    
    DEFAULT_SYSTEM_PROMPT = get_sys_prompt()

    procssed = set()
    
    for i, d in enumerate(zip(goals, targets)):
        if i < args.start or i > args.end:
            continue
        
        print(f"\n[Query]: {i}/{len(data)}")
        csv_path = os.path.join(args.result_path, f'prompt_{i}.csv')
        if os.path.exists(csv_path):
            print(f'[Result File]: {csv_path}')
            continue
        
        goal = d[0].strip()
        target = d[1].strip()

        x = goal.strip()
        z = target.strip()
        
        if ' '.join([x, z]) in procssed:
            continue
        procssed.add(' '.join([x, z]))

        SiG = decode(model, tokenizer, device, x, z, args, DEFAULT_SYSTEM_PROMPT, q_id=i)                 