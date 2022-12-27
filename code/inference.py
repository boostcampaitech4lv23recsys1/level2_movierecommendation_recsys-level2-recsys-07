import argparse
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    check_path,
    generate_submission_file,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

#multivae
from trainers import make_submission
from datasets import MultiVAEDataLoader, DeepfmDataset
from models import MultiVAE, DeepFM
import torch.optim as optim
import pandas as pd
import pickle

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="/opt/ml/input/code/output", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)
    parser.add_argument("--do_eval", action="store_true")

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    #multivae
    parser.add_argument('--model', default='deepfm', type=str) # multivae 추가
    parser.add_argument('--data', type=str, default='/opt/ml/input/data/train/',
                        help='Movielens dataset location')

    # parser.add_argument('--lr', type=float, default=1e-4,
    #                     help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    # parser.add_argument('--batch_size', type=int, default=500,
    #                     help='batch size')
    # parser.add_argument('--epochs', type=int, default=20,
    #                     help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    # parser.add_argument('--seed', type=int, default=1111,
    #                     help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='/opt/ml/input/code/output/multivae.pt',
                        help='path to save the final model')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda


    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'multivae':
        # Load the best saved model.
        with open('/opt/ml/input/code/output/multivae.pt', 'rb') as f:
            model = torch.load(f)

        loader = MultiVAEDataLoader(args.data)

        test_data = loader.load_data('test')

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
        criterion = MultiVAE.loss_function_vae
        N = test_data.shape[0]
        # Run on test data.
        final = make_submission(model = model, criterion = criterion, data_tr = test_data, is_VAE=True, batch_size = args.batch_size, N = N, device = device, total_anneal_steps = args.total_anneal_steps, anneal_cap = args.anneal_cap)
        print('=' * 89)
        print(final)
        print(len(final))

#셔플 안할거면 이거 
        # tr_ = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
        # u_list = list(tr_['user'].unique())
        u_list = list(range(0,31360))
        arr = []
        for i in u_list:
            for _ in range(10):
                arr.append(i)
        len(arr)
        final['user'] = arr

        with open('/opt/ml/input/data/train/pro_sg/show2id_multivae.pickle', 'rb') as fr:
            aa = pickle.load(fr)

        final = final.rename(columns = {0:'item'})

        ee = final['item'].apply(lambda x: aa[x])

        final['item'] = ee

#셔플 안할거면 이거 근데 셔플 안한게 성능이 더 높네 섞나안섞나 차이 거의 없는듯?
        with open('/opt/ml/input/data/train/pro_sg/profile2id_multivae.pickle', 'rb') as fr:
            aa = pickle.load(fr)

        ee = final['user'].apply(lambda x: aa[x])

        final['user'] = ee

        final = final[['user','item']]

        final = final.sort_values(by='user', ascending=True)

        final.to_csv('/opt/ml/input/code/output/submission_multivae.csv')

    elif args.model =='deepfm':
        print('deepfm')

        data = pd.read_csv('/opt/ml/input/data/train/mission3_data_3397.csv')
        n_data = len(data)
        n_user = len(data['user'].unique())
        n_item = len(data['item'].unique())
        n_genre = len(data['genre'].unique())
        offsets = [0, n_user, n_user+n_item]

        #6. feature matrix X, label tensor y 생성
        user_col = torch.tensor(data.loc[:,'user'])
        item_col = torch.tensor(data.loc[:,'item'])
        genre_col = torch.tensor(data.loc[:,'genre'])

        for col, offset in zip([user_col, item_col, genre_col], offsets):
            col += offset


        X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
        y = torch.tensor(list(data.loc[:,'rating']))
        #inference - inference
        dataset = DeepfmDataset(X, y)
        
        with open('/opt/ml/input/code/output/deepfm.pt', 'rb') as f:
            model = torch.load(f)

        model.eval()

        all_data_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        final_list = []

        with torch.no_grad():

            for x, y in all_data_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                output = output.cpu().numpy()
                final_list.append(pd.DataFrame(output.reshape(-1,1)))

        final = pd.concat(final_list)
        final = final.reset_index(drop = True)


        df = data.copy()
        df = df.drop(['Unnamed: 0', 'rating','genre'], axis=1)

        with open('/opt/ml/input/data/train/pro_sg/users_dict_reverse_deepfm.pickle', 'rb') as fr:
            aa = pickle.load(fr)

        ee = df['user'].apply(lambda x: aa[x])
        df['user'] = ee

        with open('/opt/ml/input/data/train/pro_sg/items_dict_reverse_deepfm.pickle', 'rb') as fr:
            aa = pickle.load(fr)

        ee = df['item'].apply(lambda x: aa[x])
        df['item'] = ee

        train = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
        train = train.drop('time', axis = 1)

        total = pd.merge(df, train, how='outer', indicator = True)
        total = total.query('_merge == "left_only"').drop(columns=['_merge'])
        total['rating'] = final
        total = total.sort_values(by='rating',ascending=False).groupby('user').head(10) #
        total = total.sort_values(by='user')
        total = total.drop('rating',axis=1)
        total = total.reset_index(drop=True)
        total.to_csv('/opt/ml/input/code/output/submission_deepfm.csv')

    else:
        args.data_file = args.data_dir + "train_ratings.csv"
        item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

        user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs(args.data_file)

        item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1

        # save model args
        args_str = f"{args.model_name}-{args.data_name}"

        print(str(args))

        args.item2attribute = item2attribute

        args.train_matrix = submission_rating_matrix

        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
        submission_sampler = SequentialSampler(submission_dataset)
        submission_dataloader = DataLoader(
            submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
        )

        model = S3RecModel(args=args)

        trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)

        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for submission!")
        preds = trainer.submission(0)

        generate_submission_file(args.data_file, preds)



if __name__ == "__main__":
    main()
