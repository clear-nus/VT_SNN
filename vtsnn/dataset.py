"""Datasets definitions for VT-SNN.

Each dataset should return an output of shape (data, target, label).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class ViTacDataset(Dataset):
    def __init__(self, path, sample_file, output_size, mode, spiking, rectangular=False):
        self.path = path
        self.output_size = output_size
        self.mode = mode
        self.rectangular = rectangular
        self.spiking = spiking
        self.samples = np.loadtxt(Path(path) / sample_file).astype("int")

        if mode != "vis":       # includes tactile
            if rectangular:
                self.right_tact = torch.load(Path(path) / "tac_right.pt")
                self.left_tact = torch.load(Path(path) / "tac_left.pt")
            else:
                tact = torch.load(Path(path) / "tact.pt")
                self.tact = tact.reshape(tact.shape[0], -1, 1, 1, tact.shape[-1])

        if mode != "tact":      # includes vision
            if spiking:         # Load the correct downsampled data
                self.vis = torch.load(Path(path) / "ds_vis.pt")
            else:
                self.vis = torch.load(Path(path) / "ds_vis_non_spike.pt")

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1

        if self.mode == "tact":
            if self.rectangular:
                inputs = [self.right_tact[input_index], self.left_tact[input_index]]
            else:
                inputs = [self.tact[input_index]]
        elif self.mode == "vis":
            inputs = [self.vis[input_index]]
        elif self.mode == "mm":
            if self.rectangular:
                inputs = [self.right_tact[input_index], self.left_tact[input_index], self.vis[input_index]]
            else:
                inputs = [self.tact[input_index], self.vis[input_index]]

        return (
            *inputs,
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]
