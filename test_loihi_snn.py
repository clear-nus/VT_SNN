import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
from snn_models.baseline_snn import SlayerLoihiMLP
from collections import OrderedDict
from slayerSNN import loihi as spikeLayer
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()

parser = argparse.ArgumentParser("Train model.")
parser.add_argument("--epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument("--network", type=str, help="Path to network.yml", required=True)
parser.add_argument(
    "--sample_file", type=int, help="Sample number to train from.", required=True
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)
parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)

args = parser.parse_args()
params = snn.params(args.network)
input_size = 156  # Tact
output_size = 2

device = torch.device("cuda")
net = SlayerLoihiMLP(
    params, input_size, args.hidden_size, output_size, quantize=False
).to(device)
net.load_state_dict(torch.load("weights-{args.epochs}.pt"))
net.fc1.weight.data = snn.utils.quantize(net.fc1.weight, 2)
net.fc2.weight.data = snn.utils.quantize(net.fc2.weight, 2)

test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=output_size,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)

correct = 0
num_samples = 0
net.eval()
with torch.no_grad():
    for i, (tact, _, target, label) in enumerate(test_loader):
        tact = tact.to(device)
        target = target.to(device)
        output = net.forward(tact)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        num_samples += len(label)

print(f"Number of Samples: {num_samples}\t Accuracy: {correct/num_samples}")
