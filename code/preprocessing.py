import pandas as pd

#multivate 추가
import argparse
import numpy as np
import os
import pickle
import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='deepfm', type=str)
    parser.add_argument('--data', type=str, default='/opt/ml/input/data/train/',
                        help='Movielens dataset location')
    args = parser.parse_args()

    if args.model == 'multivae':

        def get_count(tp, id):
            playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
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
        vd_users = unique_uid[(n_users - n_heldout_users * 2):]
        te_users = unique_uid[:]

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




        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

        test_data = numerize(test_plays, profile2id, show2id)
        test_data.to_csv(os.path.join(pro_dir, 'test_multivae.csv'), index=False)


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


    elif args.model == 'deepfm':

        # 1. Rating df 생성
        rating_data = "/opt/ml/input/data/train/train_ratings.csv"

        raw_rating_df = pd.read_csv(rating_data)


        b = raw_rating_df.groupby('item').count()
        b = b[b['user'] > 197]
        b = b[:]   #이걸로 네거티브 샘플링 갯수 조절
        movies_list = list(b.index)

        raw_rating_df['rating'] = 1.0 # implicit feedback
        raw_rating_df.drop(['time'],axis=1,inplace=True)
        print("Raw rating df")
        print(raw_rating_df)

        users = set(raw_rating_df.loc[:, 'user'])
        items = set(raw_rating_df.loc[:, 'item'])

        #2. Genre df 생성
        genre_data = "/opt/ml/input/data/train/genres.tsv"

        raw_genre_df = pd.read_csv(genre_data, sep='\t')

        raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

        genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}

        raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경
        print("Raw genre df - changed to id")
        print(raw_genre_df)

        #2. writer df 생성
        writer_data = "/opt/ml/input/data/train/writers.tsv"
        raw_writer_df = pd.read_csv(writer_data, sep='\t')

        raw_writer_df = raw_writer_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

        writer_dict = {writer:i for i, writer in enumerate(set(raw_writer_df['writer']))}

        raw_writer_df['writer']  = raw_writer_df['writer'].map(lambda x : writer_dict[x]) #writer id로 변경
        print("Raw writer df - changed to id")
        print(raw_writer_df)

        #2. years df 생성
        years_data = "/opt/ml/input/data/train/years.tsv"
        raw_years_df = pd.read_csv(years_data, sep='\t')

        raw_years_df = raw_years_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

        years_dict = {years:i for i, years in enumerate(set(raw_years_df['year']))}

        raw_years_df['year']  = raw_years_df['year'].map(lambda x : years_dict[x]) #writer id로 변경
        print("Raw years df - changed to id")
        print(raw_years_df)

        #2. director df 생성
        director_data = "/opt/ml/input/data/train/directors.tsv"
        raw_director_df = pd.read_csv(director_data, sep='\t')

        raw_director_df = raw_director_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

        director_dict = {director:i for i, director in enumerate(set(raw_director_df['director']))}

        raw_director_df['director']  = raw_director_df['director'].map(lambda x : director_dict[x]) #writer id로 변경
        print("Raw director df - changed to id")
        print(raw_director_df)

        #2. title df 생성
        title_data = "/opt/ml/input/data/train/titles.tsv"
        raw_title_df = pd.read_csv(title_data, sep='\t')

        raw_title_df = raw_title_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

        title_dict = {title:i for i, title in enumerate(set(raw_title_df['title']))}

        raw_title_df['title']  = raw_title_df['title'].map(lambda x : title_dict[x]) #writer id로 변경
        print("Raw titler df - changed to id")
        print(raw_title_df)

        # 3. Negative instance 생성
        print("Create Nagetive instances")
        user_group_dfs = list(raw_rating_df.groupby('user')['item'])

        first_row = True
        user_neg_dfs = pd.DataFrame()
        for u, u_items in user_group_dfs:
            u_items = set(u_items)
            i_user_neg_item = [i for i in movies_list if i not in u_items] #이거 끄면됨#b
            i_user_neg_item = np.array(i_user_neg_item) #이거 끄면됨#b
            num_negative = len(i_user_neg_item) #이거 끄면됨#b
            
            i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
            if first_row == True:
                user_neg_dfs = i_user_neg_df
                first_row = False
            else:
                user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

        raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)

        # 4. Join dfs
        joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')
        print("Joined rating df")
        print(joined_rating_df)

        # 5. user, item을 zero-based index로 mapping
        users = list(set(joined_rating_df.loc[:,'user']))
        users.sort()
        items =  list(set((joined_rating_df.loc[:, 'item'])))
        items.sort()
        genres =  list(set((joined_rating_df.loc[:, 'genre'])))
        genres.sort()

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            users_dict_reverse = {i : users[i] for i in range(len(users))}
            joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
            users = list(set(joined_rating_df.loc[:,'user']))
            
        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            items_dict_reverse = {i : items[i] for i in range(len(items))}
            joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
            items =  list(set((joined_rating_df.loc[:, 'item'])))

        joined_rating_df = joined_rating_df.sort_values(by=['user'])
        joined_rating_df.reset_index(drop=True, inplace=True)

        data = joined_rating_df
        print("Data")
        print(data)

        n_data = len(data)
        n_user = len(users)
        n_item = len(items)
        n_genre = len(genres)

        print("# of data : {}\n# of users : {}\n# of items : {}\n# of genres : {}".format(n_data, n_user, n_item, n_genre))

        data.to_csv('/opt/ml/input/data/train/deepfm_data.csv')

        with open ('/opt/ml/input/data/train/pro_sg/users_dict_reverse_deepfm.pickle', 'wb') as fw:
            pickle.dump(users_dict_reverse, fw)
        with open ('/opt/ml/input/data/train/pro_sg/items_dict_reverse_deepfm.pickle', 'wb') as fw:
            pickle.dump(items_dict_reverse, fw)

    else:

        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        array, index = pd.factorize(genres_df["genre"])
        genres_df["genre"] = array
        genres_df.groupby("item")["genre"].apply(list).to_json(
            "data/Ml_item2attributes.json"
        )


if __name__ == "__main__":
    main()
