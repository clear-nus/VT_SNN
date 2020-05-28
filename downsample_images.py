import argparse
import torch
import numpy as np
from torch import nn
from pathlib import Path
import slayerSNN as snn

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="Path", required=True)
parser.add_argument("--count", type=int, help="count", required=True)
parser.add_argument("--theta", type=float, help="SRM threshold.", required=True)
parser.add_argument("--tauRho", type=float, help="spike pdf parameter.", required=True)
parser.add_argument("--tsample", type=int, help="tSample", required=True)

args = parser.parse_args()

params = {
    "neuron": {
        "type": "SRMALPHA",
        "theta": args.theta,
        "tauSr": 10.0,
        "tauRef": 1.0,
        "scaleRef": 2,
        "tauRho": args.tauRho, # pdf
        "scaleRho": 1,
    },
    "simulation": {"Ts": 1.0, "tSample": args.tsample, "nSample": 1}
}

# define average pooling for vision
class SumPool(nn.Module):

    def __init__(self, netParams):
        super(SumPool, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.pool = self.slayer.pool(4)

    def forward(self, input_data):
        spike_out = self.slayer.spike(self.pool(self.slayer.psp(input_data)))
        return spike_out

device = torch.device('cuda:1')

    
net = SumPool(params).to(device)

tact_arr = []

print('Starting tactile ...')

for i in range(args.count):
    print(f"Processing tactile {i}...")
    # tactile
    tact_npy = Path(args.path) / f"{i}_tact.npy"
    tact = torch.FloatTensor(np.load(tact_npy))
    tact_arr.append(tact)
    
tact = torch.stack(tact_arr)
print(f"tact: {tact.shape}")
torch.save(tact, Path(args.path) / "tact.pt")

del tact

ds_vis_arr = []

print('Starting vision ...')

for i in range(args.count):
   
    print(f"Processing  {i}...")
    vis_npy = Path(args.path) / f"{i}_vis.npy"
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    vis = vis.to(device)
   
    with torch.no_grad():
        vis_pooled = net.forward(vis)
        
    ds_vis_arr.append(vis_pooled.detach().cpu().squeeze(0))

ds_vis = torch.stack(ds_vis_arr)
print(f"ds_vis: {ds_vis.shape}")
torch.save(ds_vis, Path(args.path) / "ds_vis.pt")

print("DONE")