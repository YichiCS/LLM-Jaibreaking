import os
import argparse
import pandas as pd

from utils import seed_everything


def get_json_data(path, start, end):
    
    data = []
    json_data = []
    for i in range(start, end):
        try:
            _path = os.path.join(path, f'prompt_{i}.csv')
            data.append(pd.read_csv(_path))
        except:
            print(f'Not exist prompt_{i}.csv')
    
    data = pd.concat(data, ignore_index=True)
    data = data[data['result'] == True]
    for index, row in data.iterrows():
        _row = {
            'id': index,
            'prompt': row['prompt'],
            'suffix': row['suffix'],
        }
        json_data.append(_row)

    json_data = pd.DataFrame(json_data)
    
    return json_data, len(data)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--exp_name", type=str, default='uc_advbench0050')
    
    parser.add_argument("--data_root", type=str, default='./data/train')
    parser.add_argument("--root", type=str, default='./results/uc_advbench0050/evaluate')
    parser.add_argument("--train_s", type=int, default=0)
    parser.add_argument("--train_e", type=int, default=50)
    parser.add_argument("--test_s", type=int, default=0)
    parser.add_argument("--test_e", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.seed != -1:
        seed_everything(args.seed)
        print(f"[Seed]: {args.seed}")
    
    os.makedirs(args.data_root, exist_ok=True)
    save_root = os.path.join(args.data_root, args.exp_name)
    os.makedirs(save_root, exist_ok=True)
    
    train_json, len_train = get_json_data(args.root, args.train_s, args.train_e)
    test_json, len_test = get_json_data(args.root, args.test_s, args.test_e)
    
    train_json.to_json(os.path.join(save_root, f'train.json'), orient='records', indent=4)
    test_json.to_json(os.path.join(save_root, f'test.json'), orient='records', indent=4)
    
    print(f'[Train Split]: {len_train} Samples @ {os.path.join(save_root, f'train.json')}')
    print(f'[Test Split]: {len_test} Samples @ {os.path.join(save_root, f'test.json')}')


if __name__ == '__main__':
    main()