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

NAME = ['1-MA', '2-FT', '3-ZS', '4-EP', '5-SR', '6-JT', '7-JT4', '8-JT3']
TYPE = ['train', 'val', 'test']

na = NAME[7]
tp = TYPE[1]


#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)
parser.add_argument('--dataset', default=na,
                    help='MA, FT, SR, EP, ZS')    

args = parser.parse_args()

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = BASE + 'datasets/'+args.dataset+'/'

dset_train = GenDataset(
        name = na + '-' + tp, 
        data_dir = data_set+tp+'/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

# dset_train = TrajectoryDataset(
#         data_set+'train/',
#         obs_len=obs_seq_len,
#         pred_len=pred_seq_len,
#         skip=1,norm_lap_matr=True)