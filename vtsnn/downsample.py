import argparse
import torch
import numpy as np
from pathlib import Path
import slayerSNN as snn
import torch.nn.functional as F
import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path", help="Path to post-processed data.", required=True
)
parser.add_argument(
    "--network_config", type=str, help="Configuration to use.", required=True
)

args = parser.parse_args()

# define average pooling for vision
class SumPool(torch.nn.Module):  # outputs spike trains
    def __init__(self, params):
        super(SumPool, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.pool = self.slayer.pool(4)

    def forward(self, input_data):
        spike_out = self.slayer.spike(self.pool(self.slayer.psp(input_data)))
        return spike_out


class AvgPool(torch.nn.Module):  # outputs continious time signal
    def __init__(self):
        super(AvgPool, self).__init__()
        self.pool = torch.nn.AvgPool3d(
            (1, 4, 4), padding=[0, 1, 1], stride=(1, 4, 4)
        )

    def forward(self, input_data):
        out_data = F.relu(self.pool(input_data))
        return out_data


device = torch.device("cuda")

net_params = snn.params(args.network_config)
net = SumPool(net_params).to(device)
net2 = AvgPool().to(device)

tact_arr = []

tact_count = len(glob.glob(str(Path(args.path) / "*_tact.npy")))
print(f"Processing {tact_count} tactile files...")

for i in range(tact_count):
    print(f"Processing tactile {i}...")
    # tactile
    tact_npy = Path(args.path) / f"{i}_tact.npy"
    tact = torch.FloatTensor(np.load(tact_npy))
    tact_arr.append(tact)

tact = torch.stack(tact_arr)
torch.save(tact, Path(args.path) / "tact.pt")

print(f"tact shape: {tact.shape}")
print("Done processing tactile.")
del tact

ds_vis_spike_arr = []
ds_vis_non_spike_arr = []

vis_count = len(glob.glob(str(Path(args.path) / "*_vis.npy")))
print(f"Processing {vis_count} vision files...")

for i in range(vis_count):

    print(f"Processing vision {i}...")
    vis_npy = Path(args.path) / f"{i}_vis.npy"
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    vis = vis.to(device)

    with torch.no_grad():
        vis_pooled_spike = net.forward(vis)
        vis_pooled_non_spike = net2.forward(vis.permute(0, 1, 4, 2, 3))

    ds_vis_spike_arr.append(vis_pooled_spike.detach().cpu().squeeze(0))
    ds_vis_non_spike_arr.append(
        vis_pooled_non_spike.squeeze(0).permute(0, 2, 3, 1).detach().cpu()
    )

ds_vis_spike = torch.stack(ds_vis_spike_arr)
print(f"ds_vis: {ds_vis_spike.shape}")
torch.save(ds_vis_spike, Path(args.path) / "ds_vis.pt")
del ds_vis_spike, ds_vis_spike_arr

ds_vis_non_spike = torch.stack(ds_vis_non_spike_arr)
print(f"ds_vis_non_spike: {ds_vis_non_spike.shape}")
torch.save(ds_vis_non_spike, Path(args.path) / "ds_vis_non_spike.pt")

print("DONE")
