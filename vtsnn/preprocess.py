""" VT-SNN data preprocessor.

Usage (from root directory):

1. With guild:

guild run preprocess save_path=/path/to/save \
  data_path=/path/to/data \

2. With plain Python:

python vtsnn/preprocess.py \
  --save_path /path/to/save \
  --data_path /path/to/data \
  --threshold 1 \
  --selection grasp_lift_hold \
  --bin_duration 0.02 \
  --n_sample_per_object 20 \
  --slip 0
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import slayerSNN as snn
import torch.nn.functional as F
import glob

import pandas as pd
import numpy as np
import os
import logging
import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from pathlib import Path

from collections import namedtuple

parser = argparse.ArgumentParser(description="VT-SNN data preprocessor.")

parser.add_argument(
    "--save_path", type=str, help="Location to save data to.", required=True
)
parser.add_argument(
    "--data_path", type=str, help="Path to dataset.", required=True
)
parser.add_argument("--blacklist_path", type=str, help="Path to blacklist.")
parser.add_argument(
    "--threshold", type=int, help="Threshold for tactile.", required=True
)
parser.add_argument(
    "--n_sample_per_object",
    type=int,
    help="Number of samples per class.",
    required=True,
)
parser.add_argument(
    "--network_config", type=str, help="Configuration to use.", required=True
)
parser.add_argument(
    "--task",
    type=str,
    help="Task to preprocess for.",
    choices=["cw", "slip"],
    required=True,
)
parser.add_argument("--seed", type=int, help="Random seed to use", default=100)

parser.add_argument(
    "--selection",
    type=str,
    help="Range of trajectory to process.",
    required=True,
)
parser.add_argument(
    "--bin_duration", type=float, help="Binning duration.", required=True
)
parser.add_argument(
    "--num_splits",
    type=int,
    help="Number of splits for stratified K-folds.",
    default=5,
)

args = parser.parse_args()

selections_cw = {  # index, offset, length
    "full": [1, 0.0, 8.5],
    "reaching": [1, 0.0, 2.0],
    "grasp": [3, 0.0, 4.0],
    "lift": [4, 0.0, 0.75],
    "hold": [4, 2.0, 1.25],
    "lift_hold": [4, 0.0, 2.0],
    "grasp_lift": [3, 0.0, 4.75],
    "grasp_lift_hold": [3, 0.0, 6.5],
}

selections_slip = {"full": [1, 0.0, 0.15]}

if args.task == "cw":
    SELECTION = selections_cw[args.selection]
    Trajectory = namedtuple(
        "Trajectory",
        ["start", "reaching", "reached", "grasping", "lifting", "holding"],
    )
elif args.task == "slip":
    SELECTION = selections_slip[args.selection]
    Trajectory = namedtuple(
        "Trajectory", ["start", "reaching", "reached", "grasping"],
    )


def read_tactile_file(data_path, obj_name):
    """Reads a tactile file from path. Returns a pandas dataframe."""
    obj_path = Path(data_path) / "aces_recordings" / f"{obj_name}.tact"
    df = pd.read_csv(
        obj_path,
        delimiter=" ",
        names=["polarity", "cell_index", "timestamp_sec", "timestamp_nsec"],
        dtype=int,
    )
    df = df.assign(timestamp=df.timestamp_sec + df.timestamp_nsec / 1000000000)
    df = df.drop(["timestamp_sec", "timestamp_nsec"], axis=1)
    return df


def read_trajectory(data_path, obj_name, start_time=None, zeroed=False):
    """Reads the trajectory from path, Returns a Trajectory."""
    obj_path = Path(data_path) / "traj_start_ends" / f"{obj_name}.startend"
    with open(obj_path, "r") as f:
        timings = list(map(float, f.read().split(" ")))
        if start_time is not None:
            delta = start_time - timings[0]
            timings = [t + delta for t in timings]
        if zeroed:
            start_time = timings[0]
            timings = [t - start_time for t in timings]
        return Trajectory(*timings)


class TactileData:
    def __init__(self, obj_name, selection):
        self.obj_name = obj_name
        self.trajectory = read_trajectory(args.data_path, obj_name)
        self.df = read_tactile_file(args.data_path, obj_name)

        traj_start, offset, self.T = selection
        self.start_t = self.trajectory[traj_start] + offset
        self.threshold = 1

    def binarize(self, bin_duration):
        bin_number = int(np.floor(self.T / bin_duration))
        data_matrix = np.zeros([80, 2, bin_number], dtype=int)

        pos_df = self.df[self.df.polarity == 1]
        neg_df = self.df[self.df.polarity == 0]

        end_t = self.start_t + bin_duration
        count = 0

        init_t = self.start_t

        while end_t <= self.T + init_t:  # start_t <= self.T
            _pos_count = pos_df[
                (
                    (pos_df.timestamp >= self.start_t)
                    & (pos_df.timestamp < end_t)
                )
            ]
            _pos_selective_cells = (
                _pos_count.cell_index.value_counts() > self.threshold
            )
            if len(_pos_selective_cells):
                data_matrix[
                    _pos_selective_cells[_pos_selective_cells].index.values - 1,
                    0,
                    count,
                ] = 1

            _neg_count = neg_df[
                (
                    (neg_df.timestamp >= self.start_t)
                    & (neg_df.timestamp < end_t)
                )
            ]
            _neg_selective_cells = (
                _neg_count.cell_index.value_counts() > self.threshold
            )
            if len(_neg_selective_cells):
                data_matrix[
                    _neg_selective_cells[_neg_selective_cells].index.values - 1,
                    1,
                    count,
                ] = 1
            self.start_t = end_t
            end_t += bin_duration
            count += 1

        data_matrix = np.delete(data_matrix, [16, 48], 0)
        return data_matrix


class CameraData:
    def __init__(self, obj_name, selection):
        # propophesee hyperparameters
        self.c = 2
        self.w = 200
        self.h = 250
        x0 = 180
        y0 = 0

        file_path = (
            Path(args.data_path) / "prophesee_recordings" / f"{obj_name}"
        )
        start_time = float(open(str(file_path) + ".start", "r").read())

        self.trajectory = read_trajectory(
            args.data_path, obj_name, start_time=start_time, zeroed=True
        )

        traj_start, offset, self.T = selection
        self.start_t = self.trajectory[traj_start] + offset

        td_data = loadmat(str(file_path) + "_td.mat")["td_data"]
        df = pd.DataFrame(columns=["x", "y", "polarity", "timestamp"])
        a = td_data["x"][0][0]
        b = td_data["y"][0][0]
        mask_x = (a >= 230) & (a < 430)
        mask_y = b >= 100
        a1 = a[mask_x & mask_y] - 230
        b1 = b[mask_x & mask_y] - 100
        df.x = a1.flatten()
        df.y = b1.flatten()
        df.polarity = td_data["p"][0][0][mask_x & mask_y].flatten()
        df.timestamp = (
            td_data["ts"][0][0][mask_x & mask_y].flatten() / 1000000.0
        )

        self.df = df
        self.threshold = 1

    def binarize(self, bin_duration):
        bin_number = int(np.floor(self.T / bin_duration))
        data_matrix = np.zeros([self.c, self.w, self.h, bin_number], dtype=int)

        pos_df = self.df[self.df.polarity == 1]
        neg_df = self.df[self.df.polarity == -1]

        end_t = self.start_t + bin_duration
        count = 0

        init_t = self.start_t

        while end_t <= self.T + init_t:  # start_t <= self.T
            _pos_count = pos_df[
                (
                    (pos_df.timestamp >= self.start_t)
                    & (pos_df.timestamp < end_t)
                )
            ]
            b = pd.DataFrame(index=_pos_count.index)
            b = b.assign(
                xy=_pos_count["x"].astype(str)
                + "_"
                + _pos_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(
                int
            )
            if xy.shape[0] > 0:
                data_matrix[0, xy[:, 0], xy[:, 1], count] = 1

            _neg_count = neg_df[
                (
                    (neg_df.timestamp >= self.start_t)
                    & (neg_df.timestamp < end_t)
                )
            ]
            b = pd.DataFrame(index=_neg_count.index)
            b = b.assign(
                xy=_neg_count["x"].astype(str)
                + "_"
                + _neg_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(
                int
            )
            if xy.shape[0] > 0:
                data_matrix[1, xy[:, 0], xy[:, 1], count] = 1

            self.start_t = end_t
            end_t += bin_duration
            count += 1

        data_matrix = np.swapaxes(data_matrix, 1, 2)

        return data_matrix


def tact_bin_save(file_name, overall_count, bin_duration, selection, save_path):
    tac_data = TactileData(file_name, selection)
    tacData = tac_data.binarize(bin_duration)
    f = save_path / f"{overall_count}_tact.npy"
    print(f"Writing {f}...")
    np.save(f, tacData.astype(np.uint8))


def vis_bin_save(file_name, overall_count, bin_duration, selection, save_dir):
    cam_data = CameraData(file_name, selection)
    visData = cam_data.binarize(bin_duration)
    f = save_dir / f"{overall_count}_vis.npy"
    print(f"Writing {f}...")
    np.save(f, visData.astype(np.uint8))


class ViTacData:
    def __init__(self, save_dir, list_of_objects, selection="full"):
        self.list_of_objects = list_of_objects
        self.iters = args.n_sample_per_object
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.selection = selection

    def binarize_save(self, bin_duration):
        "saves binned tactile and prophesee data"
        overall_count = 0
        big_list_tact = []
        big_list_vis = []
        for obj in self.list_of_objects:
            for i in range(1, self.iters + 1):
                file_name = f"{obj}_{i:02}"
                big_list_tact.append(
                    [
                        file_name,
                        overall_count,
                        bin_duration,
                        self.selection,
                        self.save_dir,
                    ]
                )
                big_list_vis.append(
                    [
                        file_name,
                        overall_count,
                        bin_duration,
                        self.selection,
                        self.save_dir,
                    ]
                )
                overall_count += 1

        Parallel(n_jobs=18)(delayed(tact_bin_save)(*zz) for zz in big_list_tact)
        Parallel(n_jobs=18)(delayed(vis_bin_save)(*zz) for zz in big_list_vis)


if args.task == "cw":
    list_of_objects2 = [
        "107-a_pepsi_bottle",
        "107-b_pepsi_bottle",
        "107-c_pepsi_bottle",
        "107-d_pepsi_bottle",
        "107-e_pepsi_bottle",
        "108-a_tuna_fish_can",
        "108-b_tuna_fish_can",
        "108-c_tuna_fish_can",
        "108-d_tuna_fish_can",
        "108-e_tuna_fish_can",
        "109-a_soymilk",
        "109-b_soymilk",
        "109-c_soymilk",
        "109-d_soymilk",
        "109-e_soymilk",
        "110-a_coffee_can",
        "110-b_coffee_can",
        "110-c_coffee_can",
        "110-d_coffee_can",
        "110-e_coffee_can",
    ]
elif args.task == "slip":
    list_of_objects2 = ["stable", "rotate"]

ViTac = ViTacData(Path(args.save_path), list_of_objects2, selection=SELECTION)

ViTac.binarize_save(bin_duration=args.bin_duration)


# create labels
labels = []
current_label = -1
overall_count = -1
for obj in list_of_objects2:
    current_label += 1
    for i in range(0, args.n_sample_per_object):
        overall_count += 1
        labels.append([overall_count, current_label])
labels = np.array(labels)

# stratified k fold
skf = StratifiedKFold(n_splits=args.num_splits, random_state=100, shuffle=True)
train_indices = []
test_indices = []


for train_index, test_index in skf.split(np.zeros(len(labels)), labels[:, 1]):
    train_indices.append(train_index)
    test_indices.append(test_index)

print(
    "Training size:",
    len(train_indices[0]),
    ", Testing size:",
    len(test_indices[0]),
)

for split in range(args.num_splits):
    np.savetxt(
        Path(args.save_path) / f"train_80_20_{split+1}.txt",
        np.array(labels[train_indices[split], :], dtype=int),
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        Path(args.save_path) / f"test_80_20_{split+1}.txt",
        np.array(labels[test_indices[split], :], dtype=int),
        fmt="%d",
        delimiter="\t",
    )

# Reprocess into compact .pt format


class SumPool(torch.nn.Module):  # outputs spike trains
    def __init__(self, params):
        super(SumPool, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.pool = self.slayer.pool(4)

    def forward(self, input_data):
        spike_out = self.slayer.spike(self.pool(self.slayer.psp(input_data)))
        return spike_out


class AvgPool(torch.nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.pool = torch.nn.AvgPool3d(
            (1, 4, 4), padding=[0, 1, 1], stride=(1, 4, 4)
        )

    def forward(self, input_data):
        out_data = F.relu(self.pool(input_data))
        return out_data


device = torch.device("cuda")

net_params = snn.params(args.network_config)
net = SumPool(net_params).to(device)
net2 = AvgPool().to(device)

tact_arr = []

tact_count = len(glob.glob(str(Path(args.save_path) / "*_tact.npy")))
print(f"Processing {tact_count} tactile files...")

for i in range(tact_count):
    print(f"Processing tactile {i}...")
    # tactile
    tact_npy = Path(args.save_path) / f"{i}_tact.npy"
    tact = torch.FloatTensor(np.load(tact_npy))
    tact_arr.append(tact)

tact = torch.stack(tact_arr)
torch.save(tact, Path(args.save_path) / "tact.pt")

print(f"tact shape: {tact.shape}")
print("Generating rectangular tactile data...")
# Convert tactile input into rectangular format using an assignment rule (for CNN)
N, _, _, T = tact.shape


def convert_to_rectangle(a):
    img_depth_list = []
    for i in range(a.shape[-1]):
        img_depth_list.append(
            [
                [
                    a[3, i],
                    (a[3, i] + a[10, i] + a[6, i]) / 3,
                    a[10, i],
                    (a[10, i] + a[29, i] + a[14, i]) / 3,
                    a[24, i],
                    (a[24, i] + a[29, i] + a[34, i]) / 3,
                    a[34, i],
                ],
                [
                    (a[3, i] + a[1, i] + a[6, i]) / 3,
                    a[6, i],
                    (a[6, i] + a[10, i] + a[11, i] + a[17, i] + a[19, i]) / 5,
                    a[17, i],
                    (a[24, i] + a[28, i] + a[29, i] + a[17, i] + a[19, i]) / 5,
                    a[29, i],
                    (a[29, i] + a[34, i] + a[36, i]) / 3,
                ],
                [
                    a[1, i],
                    (a[1, i] + a[6, i] + a[8, i] + a[11, i] + a[15, i]) / 5,
                    a[11, i],
                    a[19, i],
                    a[28, i],
                    (a[22, i] + a[28, i] + a[29, i] + a[36, i] + a[32, i]) / 5,
                    a[36, i],
                ],
                [
                    (a[0, i] + a[1, i] + a[5, i] + a[8, i]) / 4,
                    a[8, i],
                    a[15, i],
                    (
                        a[11, i]
                        + a[15, i]
                        + a[14, i]
                        + a[20, i]
                        + a[26, i]
                        + a[22, i]
                        + a[28, i]
                        + a[19, i]
                    )
                    / 8,
                    a[22, i],
                    a[32, i],
                    (a[32, i] + a[33, i] + a[38, i] + a[36, i]) / 4,
                ],
                [
                    a[0, i],
                    a[5, i],
                    a[14, i],
                    a[20, i],
                    a[26, i],
                    a[33, i],
                    a[38, i],
                ],
                [
                    (a[0, i] + a[2, i] + a[5, i] + a[9, i]) / 4,
                    a[9, i],
                    a[16, i],
                    (
                        a[14, i]
                        + a[16, i]
                        + a[13, i]
                        + a[21, i]
                        + a[27, i]
                        + a[23, i]
                        + a[26, i]
                        + a[20, i]
                    )
                    / 8,
                    a[23, i],
                    a[30, i],
                    (a[33, i] + a[30, i] + a[38, i] + a[37, i]) / 4,
                ],
                [
                    a[2, i],
                    (a[2, i] + a[7, i] + a[9, i] + a[13, i] + a[16, i]) / 5,
                    a[13, i],
                    a[21, i],
                    a[27, i],
                    (a[23, i] + a[27, i] + a[31, i] + a[37, i] + a[30, i]) / 5,
                    a[37, i],
                ],
                [
                    (a[2, i] + a[7, i] + a[4, i]) / 3,
                    a[7, i],
                    (a[12, i] + a[13, i] + a[7, i] + a[18, i] + a[21, i]) / 5,
                    a[18, i],
                    (a[21, i] + a[18, i] + a[25, i] + a[31, i] + a[27, i]) / 5,
                    a[31, i],
                    (a[31, i] + a[35, i] + a[37, i]) / 3,
                ],
                [
                    a[4, i],
                    (a[4, i] + a[7, i] + a[18, i]) / 3,
                    a[12, i],
                    (a[12, i] + a[18, i] + a[25, i]) / 3,
                    a[25, i],
                    (a[31, i] + a[25, i] + a[35, i]) / 3,
                    a[35, i],
                ],
            ]
        )
    return img_depth_list


all_data_l = np.zeros([N, 2, 9, 7, T])
all_data_r = np.zeros([N, 2, 9, 7, T])


for k in range(N):
    print(f"Processing tactile {k}...")
    sample_file = tact[k, ...]
    channel_list_left = []
    channel_list_right = []
    for channel in range(2):
        a = sample_file[:, channel, :]
        img_depth_right = convert_to_rectangle(a[:39])
        img_depth_left = convert_to_rectangle(a[39:])
        img_depth = img_depth_right + img_depth_left
        channel_list_left.append(img_depth_left)
        channel_list_right.append(img_depth_right)

    ri_left = np.array(channel_list_left)
    ri_left[ri_left > 0.5] = 1
    ri_left[ri_left <= 0.5] = 0
    ri_left = ri_left.swapaxes(2, 1).swapaxes(2, 3).astype(float)

    # ri_left = torch.FloatTensor(ri_left)
    all_data_l[k] = ri_left

    ri_right = np.array(channel_list_right)
    ri_right[ri_right > 0.5] = 1
    ri_right[ri_right <= 0.5] = 0

    ri_right = ri_right.swapaxes(2, 1).swapaxes(2, 3).astype(float)
    # ri_right = torch.FloatTensor(ri_left)
    all_data_r[k] = ri_right


all_data_r = torch.FloatTensor(all_data_r.astype(float))
all_data_l = torch.FloatTensor(all_data_l.astype(float))

print(f"right rectangular tactile : {all_data_r.shape}")
print(f"left rectangular tactile : {all_data_l.shape}")

torch.save(all_data_r, Path(args.save_path) / "tac_right.pt")
torch.save(all_data_l, Path(args.save_path) / "tac_left.pt")

print("Done processing tactile.")

del tact
del all_data_r
del all_data_l

ds_vis_spike_arr = []
ds_vis_non_spike_arr = []

vis_count = len(glob.glob(str(Path(args.save_path) / "*_vis.npy")))
print(f"Processing {vis_count} vision files...")

for i in range(vis_count):

    print(f"Processing vision {i}...")
    vis_npy = Path(args.save_path) / f"{i}_vis.npy"
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    vis = vis.to(device)

    with torch.no_grad():
        vis_pooled_spike = net.forward(vis)
        vis_pooled_non_spike = net2.forward(vis.permute(0, 1, 4, 2, 3))

    ds_vis_spike_arr.append(vis_pooled_spike.detach().cpu().squeeze(0))
    ds_vis_non_spike_arr.append(
        vis_pooled_non_spike.squeeze(0).permute(0, 2, 3, 1).detach().cpu()
    )

ds_vis_spike = torch.stack(ds_vis_spike_arr)
print(f"ds_vis: {ds_vis_spike.shape}")
torch.save(ds_vis_spike, Path(args.save_path) / "ds_vis.pt")
del ds_vis_spike, ds_vis_spike_arr

ds_vis_non_spike = torch.stack(ds_vis_non_spike_arr)
print(f"ds_vis_non_spike: {ds_vis_non_spike.shape}")
torch.save(ds_vis_non_spike, Path(args.save_path) / "ds_vis_non_spike.pt")
