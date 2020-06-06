import argparse
import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path", required=True)
args = parser.parse_args()

tact = torch.load(Path(args.path) / "tact.pt")
N, _,_, T = tact.shape

def convert_to_rectangle(a):
    img_depth_list = []
    for i in range(a.shape[-1]):
        img_depth_list.append([
            [a[3,i], (a[3,i]+a[10,i]+a[6,i])/3, a[10,i], (a[10,i]+a[29,i]+a[14,i])/3 , a[24,i], (a[24,i]+a[29,i]+a[34,i])/3, a[34,i]],
            [(a[3,i]+a[1,i]+a[6,i])/3   , a[6,i], (a[6,i]+a[10,i]+a[11,i]+a[17,i]+a[19,i])/5, a[17,i], (a[24,i]+a[28,i]+a[29,i]+a[17,i]+a[19,i])/5 , a[29,i], (a[29,i]+a[34,i]+a[36,i])/3],
            [a[1,i], (a[1,i]+a[6,i]+a[8,i]+a[11,i]+a[15,i])/5, a[11,i], a[19,i], a[28,i], (a[22,i]+a[28,i]+a[29,i]+a[36,i]+a[32,i])/5, a[36,i]],
            [(a[0,i]+a[1,i]+a[5,i]+a[8,i])/4, a[8,i], a[15,i], (a[11,i]+a[15,i]+a[14,i]+a[20,i]+a[26,i]+a[22,i]+a[28,i]+a[19,i])/8, a[22,i], a[32,i], (a[32,i]+a[33,i]+a[38,i]+a[36,i])/4],
            [a[0,i], a[5,i], a[14,i], a[20,i], a[26,i], a[33,i], a[38,i]],
            [(a[0,i]+a[2,i]+a[5,i]+a[9,i])/4, a[9,i], a[16,i], (a[14,i]+a[16,i]+a[13,i]+a[21,i]+a[27,i]+a[23,i]+a[26,i]+a[20,i])/8, a[23,i], a[30,i], (a[33,i]+a[30,i]+a[38,i]+a[37,i])/4],
            [a[2,i], (a[2,i]+a[7,i]+a[9,i]+a[13,i]+a[16,i])/5, a[13,i], a[21,i], a[27,i], (a[23,i]+a[27,i]+a[31,i]+a[37,i]+a[30,i])/5, a[37,i]],
            [(a[2,i]+a[7,i]+a[4,i])/3 , a[7,i], (a[12,i]+a[13,i]+a[7,i]+a[18,i]+a[21,i])/5, a[18,i], (a[21,i]+a[18,i]+a[25,i]+a[31,i]+a[27,i])/5, a[31,i], (a[31,i]+a[35,i]+a[37,i])/3],
            [a[4,i], (a[4,i]+a[7,i]+a[18,i])/3   , a[12,i], (a[12,i]+a[18,i]+a[25,i])/3, a[25,i], (a[31,i]+a[25,i]+a[35,i])/3, a[35,i]]          
        ])
    return img_depth_list


all_data_l = np.zeros([N, 2, 9, 7, T])
all_data_r = np.zeros([N, 2, 9, 7, T])


for k in range(N):
    print(f"Processing tactile {k}...")
    sample_file = tact[k,...]
    channel_list_left = []
    channel_list_right = []
    for channel in range(2):
        a = sample_file[:,channel,:]
        img_depth_right = convert_to_rectangle(a[:39])
        img_depth_left = convert_to_rectangle(a[39:])
        img_depth = img_depth_right + img_depth_left
        channel_list_left.append(img_depth_left)
        channel_list_right.append(img_depth_right)
        
    ri_left = np.array(channel_list_left)
    ri_left[ri_left > 0.5] = 1
    ri_left[ri_left <= 0.5] = 0
    ri_left = ri_left.swapaxes(2,1).swapaxes(2,3).astype(float)
    
    #ri_left = torch.FloatTensor(ri_left)
    all_data_l[k] = ri_left
    
    ri_right = np.array(channel_list_right)
    ri_right[ri_right > 0.5] = 1
    ri_right[ri_right <= 0.5] = 0
    
    ri_right = ri_right.swapaxes(2,1).swapaxes(2,3).astype(float)
    #ri_right = torch.FloatTensor(ri_left)
    all_data_r[k] = ri_right


all_data_r = torch.FloatTensor( all_data_r.astype(float) )
all_data_l = torch.FloatTensor( all_data_l.astype(float) )

print(f"right rectangular tactile : {all_data_r.shape}")
print(f"left rectangular tactile : {all_data_l.shape}")

torch.save(all_data_r, Path(args.path) / 'tac_right.pt')
torch.save(all_data_l, Path(args.path) / 'tac_left.pt')