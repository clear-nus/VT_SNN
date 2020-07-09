#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from pathlib import Path
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.animation as animation
import argparse
import scipy.io as sio


parser = argparse.ArgumentParser(description="Spike video generation.")
parser.add_argument(
    "--data_dir", type=str, help="Location to load data to.", required=True
)
parser.add_argument(
    "--target_dir", type=str, help="Location to save data to.", required=True
)
parser.add_argument(
    "--weight_class", type=str, help="Weight index for filename", required=True
)
parser.add_argument(
    "--obj", type=str, help="Filename to be saved.", required=True
)

#pepsi, tuna, soy, coffee
list_of_items = {
    'pepsi':[0, 20, 40, 60, 80],
    'tuna':[100, 120, 140, 160, 180],
    'soy':[200, 220, 240, 260, 280],
    'coffee':[300, 320, 340, 360, 380],
}
mappings = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}

args = parser.parse_args()

iter_num = list_of_items[args.obj][mappings[args.weight_class]]

# target_dir = 'vision/'
# iter_num = 0
# fname='pepsi'
# weight_class='a'

# # please indicate data dir
# data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN_new/'
train_dataset = torch.load(Path(args.data_dir) / "ds_vis.pt")



def get_data(num):
    # this function outputs input tactile, vision and downsampled vision data
    vis = torch.FloatTensor(np.load(Path(args.data_dir) / f'{num:d}_vis.npy'))
    tac = torch.FloatTensor(np.load(Path(args.data_dir) / f'{num:d}_tact.npy'))
    return tac, vis, train_dataset[num]

taxel_locations = np.asarray(sio.loadmat('/home/tasbolat/some_python_examples/VT_SNN/utility_files/finger_lut.mat')['L2'])

def resize_taxel_locs(taxel_locations, kernel_size=5):
    H,W = taxel_locations.shape
    x = []
    for i in range(0,H-kernel_size, kernel_size):
        y = []
        for j in range(0, W-kernel_size, kernel_size):
            temp_matrix = taxel_locations[i:i+kernel_size, j:j+kernel_size]
            taxel_id = np.argmax(np.bincount(temp_matrix.flatten()))
            y.append(taxel_id)
        x.append(y)
    return np.array(x).astype(np.uint8)

taxel_locations = resize_taxel_locs(taxel_locations)
taxel_size = np.zeros(39)
for taxel_loc in range(39):
    taxel_size[taxel_loc] = len(taxel_locations[taxel_locations==taxel_loc+1])

    

def generate_tact_frames(spike_data, name='temp', finger='left', resize=None):
    W,C,T = spike_data.size()
    
    red = np.repeat(taxel_locations, 325).reshape(taxel_locations.shape+(325,))
    green = np.zeros((1,)+taxel_locations.shape+(T,))
    blue = red.copy()
    
    if finger == 'left':
        spike_data=spike_data[:39,...]
    else:
        spike_data=spike_data[39:,...]
    
    for taxel_loc in range(39):
        red[red==taxel_loc+1] = spike_data[taxel_loc,0].repeat(int(taxel_size[taxel_loc]))
        blue[blue==taxel_loc+1] = spike_data[taxel_loc,1].repeat(int(taxel_size[taxel_loc]))

    red *= 255
    blue *= 255
    red = np.expand_dims(red, axis=0)
    blue = np.expand_dims(blue, axis=0)
    
    rgb = np.concatenate([red, green, blue]).astype(np.uint8)
    gif_ims = []
    for t in range(0,T,1):
        frame = rgb[:,:,:,t].transpose(1,2,0)
        if resize!= None:
            frame = frame.resize((H*resize,W*resize), Image.ANTIALIAS)
        gif_ims.append(frame)
    return gif_ims

    
    
### Generate for each object
tac, vis, ds_vis = get_data(iter_num)

tact_right = generate_tact_frames(tac, finger='right')
tact_left = generate_tact_frames(tac, finger='left')



# set writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=20000)



fig = plt.figure()
plt.tight_layout(True)
plt.axis('Off')

def animate_right(i):
    print(i)
    plt.imshow(tact_right[i])
    

print('Generating tactile right video ...')
ani = matplotlib.animation.FuncAnimation(fig, animate_right, frames=325, repeat=True)
ani.save(Path(args.target_dir) / f'tactile_right_{args.obj}_{args.weight_class}_{iter_num:02}.mp4', writer=writer)
print('Done')

plt.close()

fig = plt.figure()
plt.tight_layout(True)
plt.axis('Off')

def animate_left(i):
    plt.imshow(tact_left[i])
    

print('Generating tactile left video ...')
ani = matplotlib.animation.FuncAnimation(fig, animate_left, frames=325, repeat=True)
ani.save(Path(args.target_dir) / f'tactile_left_{args.obj}_{args.weight_class}_{iter_num:02}.mp4', writer=writer)
print('Done')

plt.close()




