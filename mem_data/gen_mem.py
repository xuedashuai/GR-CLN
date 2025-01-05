import torch
import numpy as np
import pickle
import argparse
import json

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import * 


BASE = '/mnt/project/xueqifan/Y/'

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)
parser.add_argument('--dataset', default='5-SR',
                    help='MA, FT, SR, EP, ZS')    

args = parser.parse_args()






#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = BASE + 'datasets/'+args.dataset+'/'


dset_train = TrajectoryDataset(
        name = BASE + 'datasets/temp/' + args.dataset + '-val.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=16,
        drop_last = True)

temp = []

for cnt,batch in enumerate(loader_train): 

    if cnt > 1000:
        break
    # print(cnt)

    temp.append(batch)

save_path = BASE + 'task5_val' + '.pt'

torch.save(temp, save_path)


