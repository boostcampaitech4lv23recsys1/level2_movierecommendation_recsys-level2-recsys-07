import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, MultiVAEDataLoader, DeepfmDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

#multivate
from models import MultiVAE, DeepFM
import time
from trainers import train, evaluate
import torch.optim as optim
import torch.nn as nn
import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="/opt/ml/input/code/output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=8, help="hidden size of transformer model"
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
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
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

    parser.add_argument("--using_pretrain", action="store_true")

    # args = parser.parse_args()

    #multivae 한다고 추가
    ## 각종 파라미터 세팅
    # parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


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

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")



    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda


    #multivae 모델 불러오기 추가
    if args.model == 'multivae':
        ###############################################################################
        # Load data
        ###############################################################################

        loader = MultiVAEDataLoader(args.data)

        n_items = loader.load_n_items()
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        # test_data_tr, test_data_te = loader.load_data('test')

        N = train_data.shape[0]
        idxlist = list(range(N))

        ###############################################################################
        # Build the model
        ###############################################################################

        p_dims = [200, 600, n_items]
        model = MultiVAE(p_dims).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
        criterion = MultiVAE.loss_function_vae

        ###############################################################################
        # Training code
        ###############################################################################

        best_n100 = -np.inf
        # update_count = 0
        answer = []
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(model, idxlist, train_data, device, epoch, is_VAE=True, criterion = criterion, optimizer = optimizer, N = N, batch_size = args.batch_size, total_anneal_steps = args.total_anneal_steps, anneal_cap = args.anneal_cap, log_interval = args.log_interval)
            val_loss, n100, r20, r50 = evaluate(model = model, criterion = criterion, data_tr = vad_data_tr, data_te = vad_data_te, is_VAE=True, batch_size = args.batch_size, N = N, device = device, total_anneal_steps = args.total_anneal_steps, anneal_cap = args.anneal_cap)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                        epoch, time.time() - epoch_start_time, val_loss,
                        n100, r20, r50))
            print('-' * 89)

            n_iter = epoch * len(range(0, N, args.batch_size))


            # Save the model if the n100 is the best we've seen so far.
            if n100 > best_n100:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_n100 = n100

    elif args.model == 'deepfm':
        print('deepfm')
        data = pd.read_csv('/opt/ml/input/data/train/mission3_data_500.csv')
        n_data = len(data)
        n_user = len(data['user'].unique())
        n_item = len(data['item'].unique())
        n_genre = len(data['genre'].unique())

        #6. feature matrix X, label tensor y 생성
        user_col = torch.tensor(data.loc[:,'user'])
        item_col = torch.tensor(data.loc[:,'item'])
        genre_col = torch.tensor(data.loc[:,'genre'])

        offsets = [0, n_user, n_user+n_item]

        for col, offset in zip([user_col, item_col, genre_col], offsets):
            col += offset

        X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
        y = torch.tensor(list(data.loc[:,'rating']))

        dataset = DeepfmDataset(X, y)
        train_ratio = 0.9

        train_size = int(train_ratio * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        device = torch.device('cuda')
        input_dims = [n_user, n_item, n_genre]
        embedding_dim = 10
        model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
        bce_loss = nn.BCELoss() # Binary Cross Entropy loss
        lr, num_epochs = args.lr, args.epochs
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_cnt = 0
        best_acc = -np.inf
        for e in range(num_epochs) :
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                model.train()
                optimizer.zero_grad()
                output = model(x)
                loss = bce_loss(output, y.float())
                loss.backward()
                optimizer.step()

            correct_result_sum = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                model.eval()
                output = model(x)
                result = torch.round(output)
                correct_result_sum += (result == y).sum().float()

            acc = correct_result_sum/len(test_dataset)*100
            
            if acc > best_acc:
                
                # with open(args.save + '/deepfm.pt', 'wb') as f:
                with open('/opt/ml/input/code/output' + '/deepfm.pt', 'wb') as f:
                    torch.save(model, f)
                print('save new pt')
                best_acc = acc
            
            else:
                early_cnt += 1
                

            if early_cnt > 2:
                print('early stoping')
                break
                
            print("Acc : {:.2f}%".format(acc.item()), 'early:' ,early_cnt, 'b-acc:', best_acc)

    else:
        model = S3RecModel(args=args)

        args.data_file = args.data_dir + "train_ratings.csv"
        item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

        user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
            args.data_file
        )

        item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1

        # save model args
        args_str = f"{args.model_name}-{args.data_name}"
        args.log_file = os.path.join(args.output_dir, args_str + ".txt")
        print(str(args))

        args.item2attribute = item2attribute
        # set item score in train set to `0` in validation
        args.train_matrix = valid_rating_matrix

        # save model
        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )

        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
        )

        test_dataset = SASRecDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.batch_size
        )
        trainer = FinetuneTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, None, args
        )

        print(args.using_pretrain)
        if args.using_pretrain:
            pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
            try:
                trainer.load(pretrained_path)
                print(f"Load Checkpoint From {pretrained_path}!")

            except FileNotFoundError:
                print(f"{pretrained_path} Not Found! The Model is same as SASRec")
        else:
            print("Not using pretrained model. The Model is same as SASRec")

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)

            scores, _ = trainer.valid(epoch)

            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        print(result_info)


if __name__ == "__main__":
    main()
