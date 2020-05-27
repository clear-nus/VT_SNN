# import argparse
# import torch
# import numpy as np
# from torch import nn
# from pathlib import Path

# parser = argparse.ArgumentParser()

# parser.add_argument("--path", help="Path", required=True)
# parser.add_argument("--count", type=int, help="count", required=True)

# args = parser.parse_args()

# device = torch.device('cuda:1')

# # define average pooling for vision
# class SimplePool(nn.Module):

#     def __init__(self):
#         super(SimplePool, self).__init__()
#         self.pool = nn.AvgPool3d((1,4,4), padding=[0,1,1], stride=(1,4,4))

#     def forward(self, input_data):
#         out_data = self.pool(input_data)
#         #print(out_data.shape)
#         return out_data

    
# net = SimplePool().to(device)

# tact_arr = []
# vis_arr = []
# ds_vis_arr = []

# for i in range(args.count):
#     print(f"Processing tactile {i}...")
#     # tactile
#     tact_npy = Path(args.path) / f"{i}_tact.npy"
#     tact = torch.FloatTensor(np.load(tact_npy))
#     tact_arr.append(tact)
    
#     # vision
#     vis_npy = Path(args.path) / f"{i}_vis.npy"
#     vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
#     vis_arr.append(vis.squeeze(0))
    
# tact = torch.stack(tact_arr)
# vis = torch.stack(vis_arr)
# print(f"tact: {tact.shape}")
# print(f"vis: {vis.shape}")
# torch.save(tact, Path(args.path) / "tact.pt")
# torch.save(vis, Path(args.path) / "vis.pt")

# del tact, vis

# for i in range(args.count):
   
#     print(f"Processing vision with pooling {i}...")
#     vis_npy = Path(args.path) / f"{i}_vis.npy"
#     vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
#     vis = vis.to(device)
#     vis = vis.permute(0,1,4,2,3)
   
#     with torch.no_grad():
#         vis_pooled = net.forward(vis)
#         vis_pooled = vis_pooled.squeeze().permute(0,2,3,1)
        
#     ds_vis_arr.append(vis_pooled.detach().cpu().squeeze(0))


# ds_vis = torch.stack(ds_vis_arr)
# print(f"ds_vis: {ds_vis.shape}")
# torch.save(ds_vis, Path(args.path) / "ds_vis.pt")

# print("DONE")

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
ds_vis_arr = []

for i in range(args.count):
    print(f"Processing {i}...")
    tact_npy = Path(args.path) / f"{i}_tact.npy"
    vis_npy = Path(args.path) / f"{i}_vis.npy"
    tact = torch.FloatTensor(np.load(tact_npy))
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    tact_arr.append(tact)
    slayer = snn.layer(params["neuron"], params["simulation"])
    pool = slayer.pool(4, stride=4)
    pooled_vis = pool(vis)
    ds_vis_arr.append(pooled_vis.squeeze(0))

tact = torch.stack(tact_arr)
ds_vis = torch.stack(ds_vis_arr)

print(f"tact: {tact.shape}")
print(f"ds_vis: {ds_vis.shape}")

torch.save(tact, Path(args.path) / "tact.pt")
torch.save(ds_vis, Path(args.path) / "ds_vis.pt")

print("DONE")