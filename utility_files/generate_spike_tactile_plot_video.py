#!/usr/bin/env python
# coding: utf-8

# ratio 2 and 2.5 for soy and pepsi

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
import copy

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

# # please indicate data dir
# data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN_new/'
train_dataset = torch.load(Path(args.data_dir) / "ds_vis.pt")


def get_data(num):
    # this function outputs input tactile, vision and downsampled vision data
    vis = torch.FloatTensor(np.load(Path(args.data_dir) / f'{num:d}_vis.npy'))
    tac = torch.FloatTensor(np.load(Path(args.data_dir) / f'{num:d}_tact.npy'))
    return tac, vis, train_dataset[num]

### Generate for each object
tac, vis, ds_vis = get_data(iter_num)
ttac = tac.reshape([-1,325])

# set writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=20000)

def get_plottable_st(spike_input):
    spike_trains = []
    bbb = []
    for n in range(156):
        bbb.append([])
    for t in range(325):
        for n in range(156):
            if spike_input[n,t] != 0:
                bbb[n] += [t*0.02]
        spike_trains.append(copy.deepcopy(bbb))
    return spike_trains

spike_trains = get_plottable_st(ttac)


fig = plt.figure(figsize=(10,3))
plt.xlim([0,6.5])
plt.ylim([0,160])
plt.xlabel('Time, s')
plt.ylabel('Taxels')
plt.tight_layout(True)



def animate(i):
    plt.eventplot(spike_trains[i])

print('Generating tactile plot video ...')
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=325, repeat=True)
ani.save(Path(args.target_dir) / f'tactile_plot_{args.obj}_{args.weight_class}_{iter_num:02}.mp4', writer=writer)
print('Done')

plt.close()




