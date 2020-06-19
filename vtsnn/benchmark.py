"""Benchmarking script for GPU run.

1. With guild:
guild run benchmark

2. With vanilla Python:

python vtsnn/benchmark.py \
 --epochs 500 \
 --lr 0.001 \
 --sample_file 1 \
 --batch_size 8 \
 --network_config network_config/correct_config.yml \
 --data_dir /path/to/preprocessed \
 --hidden_size 32 \
 --loss NumSpikes \
 --mode tact \
 --task cw
"""

import argparse
import pickle
import json
import time
import numpy as np
import torch
import csv
import string

from datetime import datetime


import subprocess
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
from snn_models.baseline_snn import SlayerLoihiMLP
from slayerSNN import optimizer as optim
from slayerSNN import loihi as spikeLayer
from slayerSNN import quantizeParams as quantize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from datetime import datetime
from vtsnn.dataset import ViTacDataset
from vtsnn.snn_models.loihi import SlayerLoihiMLP

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument("--bsize", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_inf", type=int)
parser.add_argument("--log", type=str)
parser.add_argument("--network_config", type=str)
parser.add_argument("--output_size", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--sample_file", type=int)
args = parser.parse_args()


class ViTacDataset(Dataset):
    def __init__(self, path, sample_file, output_size, size):
        self.path = path
        self.output_size = output_size
        self.size = size
        sample_file = Path(path) / sample_file
        self.samples = np.loadtxt(sample_file).astype("int")
        tact = torch.load(Path(path) / "tact.pt")
        self.tact = tact.reshape(tact.shape[0], -1, 1, 1, tact.shape[-1])

    def __getitem__(self, index):
        index = index % self.samples.shape[0]
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
        return self.size


input_size = 156  # Tact

device = torch.device(f"cuda:{args.gpu}")
params = snn.params(args.network_config)
net = SlayerLoihiMLP(
    params, input_size, args.hidden_size, args.output_size, quantize=False
).to(device)
net.load_state_dict(torch.load("weights-500.pt"))
net.fc1.weight.data = snn.utils.quantize(net.fc1.weight, 2)
net.fc2.weight.data = snn.utils.quantize(net.fc2.weight, 2)

test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=args.output_size,
    size=args.num_inf,
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.bsize, shuffle=False, num_workers=4
)

proc = subprocess.Popen(
    [
        "nvidia-smi",
        "-i",
        f"{args.gpu}",
        "-f",
        "gpu.csv",
        "--loop-ms=200",
        "--format=csv",
        "--query-gpu=timestamp,power.draw",
    ]
)

time.sleep(5)

step_count = 0
start_time = time.time()
start_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for i, (tact, _, _, _) in enumerate(test_loader):
    tact = tact.to(device)
    output = net.forward(tact)

end_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
elapsed_time = time.time() - start_time

time.sleep(5)

proc.terminate()  # Terminate logging to CSV

time.sleep(5)


def format(timestamp):
    """Convert timestamp to format used by nvidia-smi, s-tui for matching"""
    timestamp = timestamp.replace("/", "-")
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.split(".")[0]

    return timestamp


summary = {}
summary["hardware"] = "GPU"
summary["start_time"] = "_".join(start_tag.split(" "))
summary["end_time"] = "_".join(end_tag.split(" "))
summary["n_seconds"] = elapsed_time
summary["n_inferences"] = args.num_inf
summary["inf_per_second"] = summary["n_inferences"] / summary["n_seconds"]
summary["batchsize"] = args.bsize
summary["log_name"] = args.log.split("/")[-1]  # use file name, not path
summary["status"] = "Running"

with open("gpu.csv", "r") as cfile:
    rows = list(csv.reader(cfile))

start = summary["start_time"]
end = summary["end_time"]

start_ids = [i for i, row in enumerate(rows) if format(row[0]) == start]
end_ids = [i for i, row in enumerate(rows) if format(row[0]) <= end]

start_idx = start_ids[0]
end_idx = end_ids[-1]

power = []
times = []

for idx, row in enumerate(rows):
    if idx >= start_idx and idx <= end_idx:
        watts = row[1]
        watts = [i for i in watts if i not in string.ascii_letters]
        watts = "".join(watts).strip()

        if len(watts) > 0:
            power.append(float(watts))
            times.append(row[0])  # timestamp is always first column

assert len(power) == len(times)

log_data = list(zip(times, power))

dt = summary["n_seconds"] / len(log_data)  # avg. dt between readings
inf_per_dt = (summary["n_inferences"] / summary["n_seconds"]) * dt

sample = summary.copy()
idle_power = 3.594056  # Previously measured
for time, watts in log_data:
    sample["timestamp"] = time
    sample["total_power"] = watts
    sample["dynamic_power"] = watts - idle_power
    sample["total_joules"] = dt * watts
    sample["dynamic_joules"] = dt * sample["dynamic_power"]
    sample["total_joules_per_inf"] = sample["total_joules"] / inf_per_dt
    sample["dynamic_joules_per_inf"] = sample["dynamic_joules"] / inf_per_dt

print(sample)

with open(args.log, "w") as jfile:
    json.dump(sample, jfile)
