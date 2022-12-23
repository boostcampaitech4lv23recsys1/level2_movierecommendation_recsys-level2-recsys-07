import os
import torch
import numpy as np
import pandas as pd

import scipy.sparse as spr
from tqdm import tqdm
from utils import write_json, load_json, debug_json
from collections import Counter
device = torch.device('cuda')

print('='*20 + " File Check " + '='*20)
if os.path.exists("movie_rec_data/mov_list.json"):
    print('File Exists')
else:
    raw_data = pd.read_csv("../data/train/train_ratings.csv")
    user_item = raw_data.groupby('user')['item'].unique()

    mov_list = []
    for user_id in user_item.index:
        mov_list.append({
                    "user": user_id,
                    "item": user_item[user_id].tolist()
                    })

    write_json(mov_list, "mov_list.json")
    print('File Creation Completed')

print('='*20 + " Preprocess " + '='*20)
movlst = pd.read_json("movie_rec_data/mov_list.json")
n_user = len(movlst)
movlst_movie = movlst['item']
movie_counter = Counter([mv for mvs in movlst_movie for mv in mvs])
movie_dict = {x: movie_counter[x] for x in movie_counter}

movie_item_iid = dict()
movie_iid_item = dict()
for i, t in enumerate(movie_dict):
  movie_item_iid[t] = i
  movie_iid_item[i] = t

n_movies = len(movie_dict)
movlst['iid'] = movlst['item'].map(lambda x: [movie_item_iid.get(s) for s in x if movie_item_iid.get(s) != None])
movlst.loc[:,'num_items'] = movlst['iid'].map(len)

row = np.repeat(range(n_user), movlst['num_items'])
col = [mv for mvs in movlst['iid'] for mv in mvs]
dat = np.repeat(1, movlst['num_items'].sum())
train_items_A = spr.csr_matrix((dat, (row, col)), shape=(n_user, n_movies)) # shape: (31360, 6807)
train_items_A = torch.Tensor(train_items_A.toarray()).to(device)
train_items_A_T = train_items_A.T.to(device)

print('='*20 + " Predict " + '='*20)
def rec(pids):
  res = []
  
  for pid in tqdm(pids):
    p = np.zeros((n_movies,1)) # shape: (6807, 1)
    p[movlst.loc[pid,'iid']] = 1 # 현재 user가 본 영화들
    p = torch.Tensor(p).to(device)


    val = (train_items_A @ p).reshape(-1)  # shape: (31360,)

    items_already = movlst.loc[pid, "iid"]

    cand_item = train_items_A_T @ val  # shape: (6807,)
    cand_item_idx = cand_item.reshape(-1).argsort()[-150:]
    cand_item_idx = torch.flip(cand_item_idx, dims=[0]).cpu().numpy()

    cand_item_idx = cand_item_idx[np.isin(cand_item_idx, items_already) == False][:10] # user가 본 적 없는 영화들
    rec_item_idx = [movie_iid_item[i] for i in cand_item_idx]

    res.append({
                "user": movlst['user'][pid],
                "item": rec_item_idx,
            })
    
  return res

answers = rec(movlst.index)

print('='*20 + " Finished! " + '='*20)
result = []
for ui in answers:
    for ii in ui['item']:
        result.append([ui['user'], ii])

result = pd.DataFrame(result).rename(columns={0:'user', 1:'item'})
result.to_csv('movie_rec_data/result.csv', index=False)