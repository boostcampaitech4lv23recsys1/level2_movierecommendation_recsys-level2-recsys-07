import sys
import os
import argparse
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.config import Config
from time import time, gmtime, strftime


if __name__ == '__main__':
    
    # begin = time.time()
    parameter_dict = {
        'train_neg_sample_args': None
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='recbole', help='name of datasets')
    parser.add_argument('--config_dir', '-cd', type=str, default='/opt/ml/input/recbole/yamls', help='configs dir')
    parser.add_argument('--config_files', '-cf',type=str,required=True, help='config files')

    args, _ = parser.parse_known_args()
    
    # config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config_file_list = os.path.join(args.config_dir, args.config_files).strip().split(' ') if args.config_files else None
    
    pre_config = Config(
        model= args.model,
        dataset= args.dataset,
        config_file_list= config_file_list,
        config_dict= parameter_dict,
    )
    
     # load model's parameter from saved best .pth
    checkpoint_dir = pre_config["checkpoint_dir"]
    file_list = os.listdir(pre_config["checkpoint_dir"])
    model_save_list = []
    for file in file_list:
        if file.startswith(pre_config["model"]):
            model_save_list.append(file)
    _saved_model_file = model_save_list[-1] if model_save_list else print("Please run model first")
    saved_model_file = os.path.join(checkpoint_dir, _saved_model_file)
    
    
    # load all from .pth
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=saved_model_file,
    ) 
    
    # load all users from raw data
    df_path = os.path.join(config['data_path'], config['dataset'])
    df = pd.read_csv(df_path + '.inter', sep='\t')
    user_list = list(map(str, df[df.columns[0]].unique()))
    uid_series = dataset.token2id(dataset.uid_field, user_list)
    
    # predict recomend items
    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    print('Predict done')
    
    
    time_now = gmtime(time())
    time_str = strftime('%Y%m%d_%I:%M:%S', time_now)
    # create submission.csv
    result = []
    for user, items in zip(user_list, external_item_list):
        for item in items:
            result.append((user, item))

    submission = pd.DataFrame(result, columns=["user", "item"])
    submission.to_csv(f"./output/{args.model}_{time_str}.csv", index=False)
    
    print(f'Save inference at ./output/{args.model}_{time_str}')