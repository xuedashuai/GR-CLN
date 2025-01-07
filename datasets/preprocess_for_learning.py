import numpy as np
import pandas as pd

NAME = [
'DR_USA_Intersection_MA',
'DR_USA_Roundabout_FT',
'DR_USA_Roundabout_SR',
'DR_USA_Roundabout_EP',
'DR_CHN_Merging_ZS']

index = 2
name = NAME[index]



# #training data
# #read the raw data
# for i in range(0,3):
#     raw_data = pd.read_csv('/mnt/project/xueqifan/dataset/interaction/recorded_trackfiles/' + name + '/vehicle_tracks_00{:.0f}.csv'.format(i))
#     #construct the datarframe for selecting features
#     df_train = pd.DataFrame(raw_data.loc[raw_data.agent_type=='car', ['track_id','timestamp_ms','x', 'y']], columns=['track_id','timestamp_ms','x', 'y'])
#     df_train = df_train.loc[:,['timestamp_ms', 'track_id', 'x', 'y']]
#     #save training data
#     df_train.to_csv('./ZS_00{:.0f}.txt'.format(i), index=False, sep='\t')


#validation data
# read the raw data
raw_data = pd.read_csv('/mnt/project/xueqifan/dataset/interaction/recorded_trackfiles/' + name + '/vehicle_tracks_003.csv')
#construct the datarframe for selecting features
df_val = pd.DataFrame(raw_data.loc[raw_data.agent_type=='car', ['track_id','timestamp_ms','x', 'y']], columns=['track_id','timestamp_ms','x', 'y'])
df_val = df_val.loc[:,['timestamp_ms', 'track_id', 'x', 'y']]
#save training data
df_val.to_csv('./val_003.txt', index=False, sep='\t')


##test data
raw_data = pd.read_csv('/mnt/project/xueqifan/dataset/interaction/recorded_trackfiles/' + name + '/vehicle_tracks_004.csv')
#construct the datarframe for selecting features
df_test = pd.DataFrame(raw_data.loc[raw_data.agent_type=='car', ['track_id','timestamp_ms','x', 'y']], columns=['track_id','timestamp_ms','x', 'y'])
# df_test = df_test.loc[0:7700,['timestamp_ms', 'track_id', 'x', 'y']]
df_test = df_test.loc[:,['timestamp_ms', 'track_id', 'x', 'y']]
#save training data
df_test.to_csv('./test_004.txt', index=False, sep='\t')
