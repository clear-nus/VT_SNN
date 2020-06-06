#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import pandas as pd



class TrajStartEnd():
    def __init__(self, obj_name, path = '/datasets/eventdata/batch2/'):
        self.path = path
        self.obj_name = obj_name
        self.obj_path = self.path + 'traj_start_ends/' + obj_name + '.startend'
        self.traj_start_end = np.array(open(self.obj_path, "r").read().split(" ")).astype(float) 


# batch2
list_of_objects2 = [
    '107-a_pepsi_bottle',
    '107-b_pepsi_bottle',
    '107-c_pepsi_bottle',
    '107-d_pepsi_bottle',
    '107-e_pepsi_bottle',
    '108-a_tuna_fish_can',
    '108-b_tuna_fish_can',
    '108-c_tuna_fish_can',
    '108-d_tuna_fish_can',
    '108-e_tuna_fish_can',
    '109-a_soymilk',
    '109-b_soymilk',
    '109-c_soymilk',
    '109-d_soymilk',
    '109-e_soymilk',
    '110-a_coffee_can',
    '110-b_coffee_can',
    '110-c_coffee_can',
    '110-d_coffee_can',
    '110-e_coffee_can'
]

for obj in list_of_objects2:
    for i in range(1,21):
        str_i = str(i)
        if i < 10:
            str_i = '0' + str_i
        obj_name = obj + "_" + str_i
        # read data
        path = '/datasets/eventdata/batch2/'
        file_path = path + "prophesee_recordings/" + obj_name
        td_data = loadmat(file_path + "_td.mat")['td_data']
        df=pd.DataFrame(columns=['x', 'y', 'polarity', 'timestamp'])
        
        a = td_data['x'][0][0]
        b = td_data['y'][0][0]

        mask_y = (b >= 100)
        mask_x = (a >= 230) & (a < 430)
        a1 = a[ mask_x & mask_y ] - 230
        b1 = b[ mask_x & mask_y ] - 100
        df.x = a1.flatten()
        df.y = b1.flatten()
        df.polarity = td_data['p'][0][0][ mask_x & mask_y ].flatten() # polarity with value -1 or 1
        df.timestamp = td_data['ts'][0][0][ mask_x & mask_y ].flatten()/1000000.0 # spiking time in microseconds, convert to seconds
#         df.timestamp = td_data['ts'][0][0].flatten()/1000000.0 # spiking time in microseconds, convert to seconds
#         df.x = td_data['x'][0][0].flatten() - x0 # x coordinate
#         df.y = td_data['y'][0][0].flatten() - y0 # y coordinate
#         df.polarity = td_data['p'][0][0].flatten() # polarity with value -1 or 1

        start_time = float(open(file_path + ".start", "r").read())
        trajStartEnd = TrajStartEnd(obj_name, path)
        delta_time = start_time - trajStartEnd.traj_start_end[0]
        trajStartEnd.traj_start_end += delta_time
        trajStartEnd.traj_start_end = trajStartEnd.traj_start_end - trajStartEnd.traj_start_end[0]

        df = df[(df.timestamp >= trajStartEnd.traj_start_end[3]) & (df.timestamp < trajStartEnd.traj_start_end[3] + 6.5)]
        df = df.reindex(columns=['x', 'y', 'timestamp', 'polarity'])
        print(df.shape)
        np.save('data02/'+obj_name+'_vis.npy', df.values)



