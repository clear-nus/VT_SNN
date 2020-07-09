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





def generate_vis_frame(spike_data, resize=True):
    C, H, W, T = spike_data.size()

    red = spike_data.numpy()[0]*255
    red = red.reshape([1,H, W,T])
    green = np.zeros([1,H, W,T])
    blue = spike_data.numpy()[1]*T
    blue = blue.reshape([1,H, W,T])
    rgb = np.concatenate([red, green, blue]).astype(np.uint8)
    gif_ims = []
    for t in range(0,T,1):
        frame = Image.fromarray(rgb[:,:,:,t].transpose(1,2,0))
        frame = frame.rotate(180)
        if resize:
            frame = frame.resize((H*4,W*4), Image.ANTIALIAS)
        gif_ims.append(frame)
    return gif_ims


### Generate for each object
tac, vis, ds_vis = get_data(iter_num) # pepsi
vision = generate_vis_frame(vis, resize=False)

fig = plt.figure()
plt.tight_layout(True)
plt.axis('Off')

def animate(i):
    plt.imshow(vision[i])

# set writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=20000)

print('Generating spike video ...')
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=325, repeat=True)
ani.save(Path(args.target_dir) / f'vision_{args.obj}_{args.weight_class}_{iter_num:02}.mp4', writer=writer)
print('Done')

plt.close()



