import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
from snn_models.baseline_snn import SlayerLoihiMLP
from slayerSNN import loihi as spikeLayer
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()

parser = argparse.ArgumentParser("Train model.")
parser.add_argument("--network", type=str, help="Path to network.yml", required=True)
parser.add_argument(
    "--saved_weights", type=str, help="Path to saved_weights.", required=True
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)

args = parser.parse_args()
params = snn.params(args.network)
input_size = 156  # Tact
output_size = 2

device = torch.device("cuda")
net = SlayerLoihiMLP(params, input_size, args.hidden_size, output_size).to(device)
net.load_state_dict(torch.load(args.saved_weights))

fc1_weights = snn.utils.quantize(net.fc1.weight, 2).cpu().data.numpy()
fc2_weights = snn.utils.quantize(net.fc2.weight, 2).cpu().data.numpy()

np.save("quantized-fc1.npy", fc1_weights)
np.save("quantized-fc2.npy", fc2_weights)

print("DONE")
