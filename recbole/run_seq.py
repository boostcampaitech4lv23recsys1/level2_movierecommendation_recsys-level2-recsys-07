import argparse
import time
from recbole.quick_start import run_recbole
import os


if __name__ == '__main__':
    begin = time.time()
    parameter_dict = {
        'train_neg_sample_args': None
        # 'gpu_id':3,
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='recbole', help='name of datasets')
    parser.add_argument('--config_dir', '-cd', type=str, default='/opt/ml/input/recbole/yamls', help='configs dir')
    parser.add_argument('--config_files', '-cf',type=str, required=True, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = os.path.join(args.config_dir, args.config_files).strip().split(' ') if args.config_files else None
    
    
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    end=time.time()
    print(end-begin)