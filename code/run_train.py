from args import parse_args
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import get_full_sort_score
from datasets import SASRecDataset, MultiVAEDataLoader
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    random_neg
)

#multivate
from models import MultiVAE
import time
from trainers import train, evaluate
import torch.optim as optim

#bert4rec
from models import BERT4Rec
from torch import nn
from preprocessing import preprocessing
from datasets import SeqDataset
import tqdm


def main():
    
    args = parse_args()
    
    
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
    
    
    if args.model == 'bert4rec':
        user_seq, user_train, user_valid, num_user, num_item, df, rating_matrix = preprocessing(args)
        valid_rating_matrix, test_rating_matrix = rating_matrix
        
        seq_dataset = SeqDataset(args, user_seq, num_user, num_item, data_type='train')
        data_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        
        model = BERT4Rec(args, num_user, num_item, device)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=50, verbose=True)
        
        num_epochs = args.epochs
        for epoch in range(1, num_epochs + 1):
            
            ## train
            model.train()
            
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            
            tbar = tqdm.tqdm(data_loader)
            for step, (log_seqs, labels) in enumerate(tbar):
                logits, _ = model(log_seqs)
                
                # size matching
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1).to(device)
                optimizer.zero_grad()
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
                tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')
            
            post_fix = {
                "Epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(tbar)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }
            
            if (epoch + 1) % args.log_freq == 0:
                print(str(post_fix))
                print()
                
            # eval
            model.eval()

            valid_dataset = SeqDataset(args, user_seq, num_user, num_item, data_type='valid')
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True)
            
            vbar = tqdm.tqdm(valid_loader)     
            
            pred_list = None
            answer_list = None
            
            for step, (user_id, log_seq, answers) in enumerate(vbar):
                with torch.no_grad():
                    _, predictions = model(log_seq)
                    predictions = - predictions
                    predictions = predictions[:, -1, :]
                    
                    # [batch hidden_size ]
                    test_item_emb = model.item_emb.weight[:6808]
                    # [batch hidden_size ]
                    rating_pred = torch.matmul(predictions, test_item_emb.transpose(0, 1))
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_id
                    rating_pred[valid_rating_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -10)[:, -10:]

                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                    batch_pred_list = ind[
                        np.arange(len(rating_pred))[:, None], arr_ind_argsort
                    ]
                    
                    if step == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(
                            answer_list, answers.cpu().data.numpy(), axis=0
                            )
                        
                    vbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Valid')
                    
            scores, _ = get_full_sort_score(epoch,answer_list, pred_list)
            early_stopping(np.array(scores[-1:]), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break



    else:
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
        
        model = S3RecModel(args=args)
        
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
