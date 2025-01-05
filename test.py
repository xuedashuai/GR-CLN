import torch
import pickle
from utils import * 

BASE = '/mnt/project/xueqifan/Y/'

name = '1-MA'
obs_seq_len = 20
pred_seq_len = 40
data_set = BASE + 'datasets/'+name+'/'

dset_train = TrajectoryDataset(
        name = BASE + 'datasets/temp/' + name + '-train.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=16,
        drop_last = True)

for cnt,batch in enumerate(loader_train): 

    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,V_obs,A_obs,V_tr,A_tr = batch

    if cnt >10:
        break
    
    print(V_obs.shape)
