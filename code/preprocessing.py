import pandas as pd

#multivate 추가
import argparse
import numpy as np
import os
import pickle

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='multivae', type=str)
    parser.add_argument('--data', type=str, default='/opt/ml/input/data/train/',
                        help='Movielens dataset location')
    args = parser.parse_args()

    if args.model == 'multivae':

        def get_count(tp, id):
            playcount_groupbyid = tp[[id]].groupby(id, as_index=True)   # pandas 옛날 버전에서 as_index=False
            count = playcount_groupbyid.size()

            return count

        # 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상) 
        # 데이터만을 추출할 때 사용하는 함수입니다.
        # 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
        def filter_triplets(tp, min_uc=5, min_sc=0):
            if min_sc > 0:
                itemcount = get_count(tp, 'item')
                tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

            if min_uc > 0:
                usercount = get_count(tp, 'user')
                tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

            usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
            return tp, usercount, itemcount

        #훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
        #100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
        #확인하기 위함입니다.
        def split_train_test_proportion(data, test_prop=0.2):
            data_grouped_by_user = data.groupby('user')
            tr_list, te_list = list(), list()

            np.random.seed(98765)
            
            for _, group in data_grouped_by_user:
                n_items_u = len(group)
                
                if n_items_u >= 5:
                    idx = np.zeros(n_items_u, dtype='bool')
                    idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                    tr_list.append(group[np.logical_not(idx)])
                    te_list.append(group[idx])
                
                else:
                    tr_list.append(group)
            
            data_tr = pd.concat(tr_list)
            data_te = pd.concat(te_list)

            return data_tr, data_te

        def numerize(tp, profile2id, show2id):
            uid = tp['user'].apply(lambda x: profile2id[x])
            sid = tp['item'].apply(lambda x: show2id[x])
            return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

        print("Load and Preprocess Movielens dataset")
        # Load Data
        DATA_DIR = args.data
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
        print("원본 데이터\n", raw_data)

        # Filter Data
        raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)
        #제공된 훈련데이터의 유저는 모두 5개 이상의 리뷰가 있습니다.
        print("5번 이상의 리뷰가 있는 유저들로만 구성된 데이터\n",raw_data)

        print("유저별 리뷰수\n",user_activity)
        print("아이템별 리뷰수\n",item_popularity)

        # Shuffle User Indices
        unique_uid = user_activity.index
        # uu = unique_uid.copy()
        print("(BEFORE) unique_uid:",unique_uid)
        np.random.seed(98765)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]
        print("(AFTER) unique_uid:",unique_uid)

        n_users = unique_uid.size #31360
        n_heldout_users = 3136


        # Split Train/Validation/Test User Indices
        tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
        vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
        te_users = unique_uid[(n_users - n_heldout_users):]
        sub_users = unique_uid[:]

        #주의: 데이터의 수가 아닌 사용자의 수입니다!
        print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
        print("검증 데이터에 사용될 사용자 수:", len(vd_users))
        print("테스트 데이터에 사용될 사용자 수:", len(te_users))


        ##훈련 데이터에 해당하는 아이템들
        #Train에는 전체 데이터를 사용합니다.
        train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

        ##아이템 ID
        unique_sid = pd.unique(train_plays['item'])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        show2id_reverse = dict((i, sid) for (i, sid) in enumerate(unique_sid))
        profile2id_reverse = dict((i, pid) for (i, pid) in enumerate(unique_uid))

        pro_dir = os.path.join(DATA_DIR, 'pro_sg')

        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
        vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

        test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]
        test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

        sub_plays = raw_data.loc[raw_data['user'].isin(sub_users)]
        sub_plays = sub_plays.loc[sub_plays['item'].isin(unique_sid)]

        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

        test_data_tr = numerize(test_plays_tr, profile2id, show2id)
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

        test_data_te = numerize(test_plays_te, profile2id, show2id)
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

        sub_data = numerize(sub_plays, profile2id, show2id)
        sub_data.to_csv(os.path.join(pro_dir, 'submit_data.csv'), index=False)

        with open ('/opt/ml/input/data/train/pro_sg/profile2id_multivae.pickle', 'wb') as fw:
            pickle.dump(profile2id_reverse, fw)

        with open ('/opt/ml/input/data/train/pro_sg/show2id_multivae.pickle', 'wb') as fw:
            pickle.dump(show2id_reverse, fw)

        print("Done!")

        #데이터 셋 확인
        print(train_data)
        print(vad_data_tr)
        print(vad_data_te)
        # print(test_data_tr)
        # print(test_data_te)

    else:

        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        array, index = pd.factorize(genres_df["genre"])
        genres_df["genre"] = array
        genres_df.groupby("item")["genre"].apply(list).to_json(
            "data/Ml_item2attributes.json"
        )


if __name__ == "__main__":
    main()
