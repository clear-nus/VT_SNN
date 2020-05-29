import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import slayerSNN as snn

# Example Usage:
# dataset = ViTacDataset(path="20200124170823_grasp_lift_hold/",
#                        sample_file="train_80_20_4.txt")


class ViTacDataset(Dataset):
    def __init__(self, path, sample_file, output_size):
        self.path = path
        self.output_size = output_size
        sample_file = Path(path) / sample_file
        self.samples = np.loadtxt(sample_file).astype("int")
        tact = torch.load(Path(path) / "tact.pt")
        self.tact = tact.reshape(tact.shape[0], -1, 1, 1, tact.shape[-1])

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1
    
        return (
            self.tact[input_index],
            torch.tensor(0),
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]


class ViTacVisDataset(Dataset):
    def __init__(self, path, sample_file, output_size):
        self.path = path
        sample_file = Path(path) / sample_file
        self.samples = np.loadtxt(sample_file).astype("int")
        self.vis = torch.load(Path(path) / "ds_vis.pt")
        self.output_size = output_size

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1

        return (
            self.vis[input_index],
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]

class ViTacMMDataset(Dataset):
    def __init__(self, path, sample_file, output_size):
        self.path = path
        tact = torch.load(Path(path) / "tact.pt")
        self.tact = tact.reshape(tact.shape[0], -1, 1, 1, tact.shape[-1])
        self.ds_vis = torch.load(Path(path) / "ds_vis.pt")
        self.samples = np.loadtxt(Path(path) / sample_file).astype("int")
        self.output_size = output_size

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1

        return (
            self.tact[input_index],
            self.ds_vis[input_index],
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]
