import argparse
import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="Path", required=True)
parser.add_argument("--count", type=int, help="count", required=True)
parser.add_argument("--network", help="network.yaml", required=True)

args = parser.parse_args()

params = snn.params(args.network)

tact_arr = []
vis_arr = []
ds_vis_arr = []

for i in range(args.count):
    tact_npy = args.path + str(i) + "_tact.npy"
    vis_npy = args.path + str(i) + "_vis.npy"
    tact = torch.FloatTensor(np.load(tact_npy))
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    tact_arr.append(tact)
    vis_arr.append(vis.squeeze(0))
    slayer = snn.layer(params["neuron"], params["simulation"])
    pool = slayer.pool(4, stride=4)
    pooled_vis = pool(vis)
    ds_vis_arr.append(pooled_vis.squeeze(0))

tact = torch.stack(tact_arr)
vis = torch.stack(vis_arr)
ds_vis = torch.stack(ds_vis_arr)

print("tact: ", tact.shape)
print("vis: ", vis.shape)
print("ds_vis: ", ds_vis.shape)

torch.save(tact, Path(args.path) / "tact.pt")
torch.save(vis, Path(args.path) / "vis.pt")
torch.save(ds_vis, Path(args.path) / "pooled_vis/ds_vis.pt")

print("DONE")