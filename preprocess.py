import pandas as pd
import numpy as np
import os
import logging
import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
from pathlib import Path

from collections import namedtuple
import argparse

np.random.seed(0)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()

Trajectory = namedtuple(
    "Trajectory", ["start", "reaching", "reached", "grasping", "lifting", "holding"]
)

selections = {  # index, offset, length
    "full": [1, 0.0, 8.5],
    "reaching": [1, 0.0, 2.0],
    "grasp": [3, 0.0, 4.0],
    "lift": [4, 0.0, 0.75],
    "hold": [4, 2.0, 1.25],
    "lift_hold": [4, 0.0, 2.0],
    "grasp_lift": [3, 0.0, 4.75],
    "grasp_lift_hold": [3, 0.0, 6.5]
}

class Modes:
    TACT, VIZ = range(2)


parser = argparse.ArgumentParser(description="Data preprocessor.")
parser.add_argument(
    "--save_dir", type=str, help="Location to save data to.", required=True
)
parser.add_argument(
    "--tactile_path", type=str, help="Path to tactile dataset.", required=True
)
parser.add_argument(
    "--video_path", type=str, help="Path to video dataset.", required=True
)
parser.add_argument(
    "--trajectory_path", type=str, help="Path to trajectories.", required=True
)
parser.add_argument("--blacklist_path", type=str, help="Path to blacklist.")
parser.add_argument(
    "--threshold", type=int, help="Threshold for tactile.", required=True
)
parser.add_argument(
    "--n_sample_per_object", type=int, help="Number of samples per class.", required=True
)


parser.add_argument(
    "--selection",
    type=str,
    choices=selections.keys(),
    help="Range of trajectory to process.",
    required=True,
)
parser.add_argument(
    "--bin_duration", type=float, help="Binning duration.", required=True
)
parser.add_argument(
    "--modes",
    type=str,
    choices=["tac", "vi", "vitac"],
    help="What files to parse.",
    default="vitac",
    required=True,
)
parser.add_argument(
    "--remove_outlier", type=bool, help="removes outliers", required=True
)
args = parser.parse_args()

def read_tactile_file(tactile_path, obj_name):
    """Reads a tactile file from path. Returns a pandas dataframe."""
    obj_path = Path(tactile_path) / f"{obj_name}.tact"
    df = pd.read_csv(
        obj_path,
        delimiter=" ",
        names=["polarity", "cell_index", "timestamp"],
        dtype={'polarity': int, 'cell_index': int, 'timestamp': float}
    )
    return df


def read_trajectory(trajectory_path, obj_name, start_time=None, zeroed=False):
    """Reads the trajectory from path, Returns a Trajectory."""
    obj_path = Path(trajectory_path) / f"{obj_name}.startend"
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
        assert selection in selections
        self.obj_name = obj_name
        self.trajectory = read_trajectory(args.trajectory_path, obj_name)
        self.df = read_tactile_file(args.tactile_path, obj_name)

        traj_start, offset, self.T = selections[selection]
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
                ((pos_df.timestamp >= self.start_t) & (pos_df.timestamp < end_t))
            ]
            _pos_selective_cells = _pos_count.cell_index.value_counts() > self.threshold
            if len(_pos_selective_cells):
                data_matrix[
                    _pos_selective_cells[_pos_selective_cells].index.values - 1,
                    0,
                    count,
                ] = 1

            _neg_count = neg_df[
                ((neg_df.timestamp >= self.start_t) & (neg_df.timestamp < end_t))
            ]
            _neg_selective_cells = _neg_count.cell_index.value_counts() > self.threshold
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
    def __init__(self, obj_name, selection="full"):
        # propophesee hyperparameters
        self.c = 2
        self.w = 200
        self.h = 250
        x0 = 180
        y0 = 0

        file_path = args.video_path + obj_name  # + ".start" # _td.mat
        start_time = float(open(file_path + ".start", "r").read())

        self.trajectory = read_trajectory(
            args.trajectory_path, obj_name, start_time=start_time, zeroed=True
        )

        traj_start, offset, self.T = selections[selection]
        self.start_t = self.trajectory[traj_start] + offset

        td_data = loadmat(file_path + "_td.mat")["td_data"]
        df = pd.DataFrame(columns=["x", "y", "polarity", "timestamp"])
        a = td_data["x"][0][0]
        b = td_data["y"][0][0]
        mask_x = (a >= 230) & (a < 430)
        mask_y = (b >= 100)
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
                ((pos_df.timestamp >= self.start_t) & (pos_df.timestamp < end_t))
            ]
            b = pd.DataFrame(index=_pos_count.index)
            b = b.assign(
                xy=_pos_count["x"].astype(str) + "_" + _pos_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(int)
            if xy.shape[0] > 0:
                data_matrix[0, xy[:, 0], xy[:, 1], count] = 1

            _neg_count = neg_df[
                ((neg_df.timestamp >= self.start_t) & (neg_df.timestamp < end_t))
            ]
            b = pd.DataFrame(index=_neg_count.index)
            b = b.assign(
                xy=_neg_count["x"].astype(str) + "_" + _neg_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(int)
            if xy.shape[0] > 0:
                data_matrix[1, xy[:, 0], xy[:, 1], count] = 1

            self.start_t = end_t
            end_t += bin_duration
            count += 1

        data_matrix = np.swapaxes(data_matrix, 1, 2)

        return data_matrix

def tact_bin_save(file_name, overall_count, bin_duration, selection, save_dir):
    tac_data = TactileData(file_name, selection)
    tacData = tac_data.binarize(bin_duration)
    f = save_dir / f"{overall_count}_tact.npy"
    log.info(f"Writing {f}...")
    np.save(f, tacData.astype(np.uint8))


def vis_bin_save(file_name, overall_count, bin_duration, selection, save_dir):
    cam_data = CameraData(file_name, selection)
    visData = cam_data.binarize(bin_duration)
    f = save_dir / f"{overall_count}_vis.npy"
    log.info(f"Writing {f}...")
    np.save(f, visData.astype(np.uint8))


class ViTacData:
    def __init__(self, save_dir, list_of_objects, selection="full"):
        self.list_of_objects = list_of_objects
        self.iters = args.n_sample_per_object
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.selection = selection

    def binarize_save(self, bin_duration, modes=[Modes.TACT, Modes.VIZ]):
        "saves binned tactile and prophesee data"
        overall_count = 0
        big_list_tact = []
        big_list_vis = []
        for obj in self.list_of_objects:
            for i in range(1, self.iters + 1):
                file_name = f"{obj}_{i:02}"
                if Modes.TACT in modes:
                    big_list_tact.append(
                        [
                            file_name,
                            overall_count,
                            bin_duration,
                            self.selection,
                            self.save_dir,
                        ]
                    )
                if Modes.VIZ in modes:
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
        if Modes.TACT in modes:
            Parallel(n_jobs=18)(delayed(tact_bin_save)(*zz) for zz in big_list_tact)
        if Modes.VIZ in modes:
            Parallel(n_jobs=18)(delayed(vis_bin_save)(*zz) for zz in big_list_vis)


# batch2
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

ViTac = ViTacData(Path(args.save_dir), list_of_objects2, selection=args.selection)

modes = [Modes.TACT]

if args.modes == "vi":
    modes = [Modes.VIZ]
elif args.modes == "vitac":
    modes = [Modes.TACT, Modes.VIZ]

ViTac.binarize_save(bin_duration=args.bin_duration, modes=modes)

# TAS edited ------------------------------------
remove_outlier = args.remove_outlier
from sklearn.model_selection import StratifiedKFold

if remove_outlier:
    path_outlier = "/home/tasbolat/some_python_examples/VT_SNN/auxillary_files/"
    remove_list = np.loadtxt(path_outlier + 'blacklisted.txt').astype(int)

# create labels
labels = []
current_label = -1
overall_count = -1
for obj in list_of_objects2:
    current_label += 1
    for i in range(0, args.n_sample_per_object):
        overall_count+=1
        if remove_outlier:
            if overall_count in remove_list[:,0]:
                continue
        labels.append([overall_count, current_label])
labels = np.array(labels)

# stratified k fold
skf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)
train_indices = []
test_indices = []


for train_index, test_index in  skf.split(np.zeros(len(labels)), labels[:,1]):
    train_indices.append(train_index)
    test_indices.append(test_index)
    #print("TRAIN:", train_index, "TEST:", test_index)

print('Training size:', len(train_indices[0]),', Testing size:',  len(test_indices[0]))

# write to the file
splits = ['80_20_1','80_20_2','80_20_3','80_20_4', '80_20_5']
count = 0
for split in splits:
    np.savetxt(args.save_dir + 'train_' + split + '.txt', np.array(labels[train_indices[count], :], dtype=int), fmt='%d', delimiter='\t')
    np.savetxt(args.save_dir + 'test_' + split + '.txt', np.array(labels[test_indices[count], :], dtype=int), fmt='%d', delimiter='\t')
    count += 1

# End of TAS edit ----------------------------------------