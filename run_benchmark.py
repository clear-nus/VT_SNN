import argparse
import pickle
import json
import time
import numpy as np
import torch

import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
from snn_models.baseline_snn import SlayerLoihiMLP
from slayerSNN import optimizer as optim
from slayerSNN import loihi as spikeLayer
from slayerSNN import quantizeParams as quantize
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

from datetime import datetime
from snn_models.baseline_snn import SlayerLoihiMLP

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument("--bsize", type=int, default=1)
parser.add_argument("--time", type=int)
parser.add_argument("--log", type=str)
parser.add_argument("--network_config", type=str)
parser.add_argument("--saved_weights", type=str)
parser.add_argument("--output_size", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--sample_file", type=int)
args = parser.parse_args()

input_size = 156  # Tact

device = torch.device("cuda:2")
params = snn.params(args.network_config)
net = SlayerLoihiMLP(params, input_size, args.hidden_size, args.output_size).to(device)
net.load_state_dict(torch.load(args.saved_weights))

test_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=args.output_size
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.bsize, shuffle=False, num_workers=4
)

time.sleep(5)  # sleep to distance from setup power consumption

step_count = 0
start_time = time.time()
start_tag = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print('Loop starting...')
while True:
    # loop over batches until time limit is reached
    for i, (tact, _, _, _) in enumerate(test_loader):
        tact = tact.to(device)
        output = net.forward(tact)
        step_count += 1

        if time.time() - start_time > args.time:
            break
    else:
        continue
    break

end_tag = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print('Elapsed time: %4f' % (time.time() - start_time))
print('Target time: %2f' % args.time)
print('Number of inferences: %d' % (step_count * args.bsize))

# write a json summmary of this benchmarking experiment
if args.log:
    summary = {}
    summary['hardware'] = "GPU"
    summary['start_time'] = '_'.join(start_tag.split(' '))
    summary['end_time'] = '_'.join(end_tag.split(' '))
    summary['n_seconds'] = args.time
    summary['n_inferences'] = step_count * args.bsize
    summary['inf_per_second'] = summary['n_inferences'] / summary['n_seconds']
    summary['batchsize'] = args.bsize
    summary['log_name'] = args.log.split('/')[-1]  # use file name, not path
    summary['status'] = 'Running'

    with open(args.log, 'w') as jfile:
        json.dump(summary, jfile)
