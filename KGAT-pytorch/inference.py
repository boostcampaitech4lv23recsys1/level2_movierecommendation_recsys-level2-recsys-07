import numpy as np
import pandas as pd
import torch
import os
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base', type=str, default='/opt/ml/input/KGAT-pytorch', help='name of datasets')
    
    parser.add_argument('--dataset', type=str, default='custom', help='name of datasets')
    
    parser.add_argument('--detail_dir', type=str,
                        default='/opt/ml/input/KGAT-pytorch/trained_model/KGAT/custom/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/cf_scores.npy',
                        help='name of datasets')

    args, _ = parser.parse_known_args()
    
    dataset_dir = os.path.join(os.path.join(args.base, 'datasets'), args.dataset)
    train = pd.read_csv(dataset_dir + '/train.csv')
    user_list = train[train.columns[0]].to_list()

    cf_score = np.load(args.detail_dir)
    topk = torch.topk(torch.tensor(cf_score), 10)
    topk_indices = np.array(topk.indices)
    
    # create submission.csv
    result = []
    for user, items in zip(user_list, topk_indices):
        for item in items:
            result.append((user, item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "KGAT-pytorch/output/inference.csv", index=False
    )
    
    print('Save inference at output/')