"""Benchmarking script for GPU run.

1. With guild:
guild run benchmark model_dir=/path/to/model \
  data_dir=/path/to/test

2. With vanilla Python:

python vtsnn/benchmark.py \
 --epochs 500 \
 --lr 0.001 \
 --sample_file 1 \
 --batch_size 8 \
 --network_config network_config/correct_config.yml \
 --model_dir /path/to/model \
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
from vtsnn.models.loihi import SlayerLoihiMLP, SlayerLoihiMM
from slayerSNN import optimizer as optim
from slayerSNN import loihi as spikeLayer
from slayerSNN import quantizeParams as quantize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from datetime import datetime
from vtsnn.dataset import ViTacDataset
from vtsnn.models.loihi import SlayerLoihiMLP, SlayerLoihiMM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", type=str, help="Path to model.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument("--bsize", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_inf", type=int)
parser.add_argument("--log", type=str)
parser.add_argument("--network_config", type=str)
parser.add_argument(
    "--task",
    type=str,
    help="The classification task.",
    choices=["slip"],  # Loihi can't handle > 16 classes
    default="slip",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["tact", "vis", "mm"],
    help="Type of model to benchmark.",
    required=True,
)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--sample_file", type=int)
args = parser.parse_args()


device = torch.device(f"cuda:{args.gpu}")
params = snn.params(args.network_config)

if args.task == "cw":
    output_size = 20
else:  # Slip
    output_size = 2

if args.mode == "tact":
    model = SlayerLoihiMLP
    model_args = {
        "params": params,
        "input_size": 156,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "vis":
    model = SlayerLoihiMLP
    model_args = {
        "params": params,
        "input_size": 50 * 63,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
else:  # NOTE: args.hidden_size unused here
    model = SlayerLoihiMM
    model_args = {
        "params": params,
        "tact_input_size": 156,
        "vis_input_size": 50 * 63,
        "tact_output_size": 50,
        "vis_output_size": 10,
        "output_size": output_size,
    }

net = model(**model_args).to(device)

net.load_state_dict(torch.load(Path(args.model_dir) / "weights_500.pt"))

if args.mode != "mm":
    net.fc1.weight.data = snn.utils.quantize(net.fc1.weight, 2)
    net.fc2.weight.data = snn.utils.quantize(net.fc2.weight, 2)
else:
    net.tact_fc.weight.data = snn.utils.quantize(net.tact_fc.weight, 2)
    net.vis_fc.weight.data = snn.utils.quantize(net.vis_fc.weight, 2)
    net.combi.weight.data = snn.utils.quantize(net.combi.weight, 2)

test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=output_size,
    spiking=True,
    loihi=True,
    mode=args.mode,
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
correct = 0
start_time = time.time()
start_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for *data, target, label in test_loader:
    data = [d.to(device) for d in data]
    target = target.to(device)
    output = net.forward(*data)
    correct += torch.sum(snn.predict.getClass(output) == label).data.item()

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
