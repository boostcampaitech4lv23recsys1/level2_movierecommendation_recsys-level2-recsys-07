import pandas as pd

#multivate 추가
import argparse
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix


def preprocessing(args, dtype='train'):
    model = args.model
    if model == 'bert4rec':
        from collections import defaultdict
        data_path = args.data + "train_ratings.csv"
        df = pd.read_csv(data_path)

        item_ids = df['item'].unique()
        user_ids = df['user'].unique()
        num_item, num_user = len(item_ids), len(user_ids)


        # user, item indexing
        item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
        user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        df.sort_values(['user_idx', 'time'], inplace=True)
        df.sort_values(['user', 'time'], inplace=True)
        # del df['item'], df['user'] 

        # train set, valid set 생성
        users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
        for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        # for u, i, t in zip(df['user'], df['item'], df['time']):
            users[u].append(i)
            
        user_seq = list(users.values())
        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_user, num_item)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_user, num_item)
        
            
        user_train = dict()
        user_valid = dict()
        user_test = dict()
        
        for user in users:
            user_train[user] = users[user][:-3]
            user_valid[user] = users[user][-2]
            user_test[user] = users[user][-1]

        # print(f'num users: {num_user}, num items: {num_item}')     
        return user_seq, user_train, user_valid, num_user, num_item, df, [valid_rating_matrix, test_rating_matrix]
    

def generate_rating_matrix_valid(user_seq, num_user, num_item):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_user, num_item + 1))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_user, num_item):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_user, num_item + 1))

    return rating_matrix