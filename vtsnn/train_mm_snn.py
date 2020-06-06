import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
#from models.slayer_multimodal import SlayerMM
from vtsnn.models.snn.multimodal_snn import SlayerMM
from torch.utils.data import DataLoader
from vtsnn.dataset import ViTacMMDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()

parser = argparse.ArgumentParser("Trainer.")
parser.add_argument("--epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir", type=str, help="Path for saving checkpoints.", required=True
)
parser.add_argument("--tsample", type=int, help="tSample", required=True)
parser.add_argument(
    "--tsr_stop", type=int, help="Target Spike Region Stop", required=True
)
parser.add_argument(
    "--tsr_start", type=int, help="Target Spike Region Start", required=True
)
parser.add_argument("--sc_true", type=int, help="Spike Count True", required=True)
parser.add_argument("--sc_false", type=int, help="Spike Count False", required=True)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file", type=int, help="Sample number to train from.", required=True
)
parser.add_argument(
    "--batch_size", type=int, help="Batch Size.", required=True
)
parser.add_argument(
    "--output_size", type=int, help="Output Size.", required=True
)
parser.add_argument(
    "--theta", type=float, help="SRM threshold.", required=True
)

parser.add_argument(
    "--tauRho", type=float, help="spike pdf parameter.", required=True
)
parser.add_argument(
    "--loss_type", type=int, help="0:numSpikes or 1:weightedNumSpikes", required=True
)
args = parser.parse_args()

LOSS_TYPES = ['NumSpikes', 'WeightedNumSpikes']

params = {
    "neuron": {
        "type": "SRMALPHA",
        "theta": args.theta,
        "tauSr": 10.0,
        "tauRef": 1.0, # 10
        "scaleRef": 2,
        "tauRho": args.tauRho, # pdf
        "scaleRho": 1,
    },
    "simulation": {"Ts": 1.0, "tSample": args.tsample, "nSample": 1},
    "training": {
        "error": {
            "type": LOSS_TYPES[args.loss_type], # "NumSpikes" or "WeightedNumSpikes"
            "tgtSpikeRegion": {  # valid for NumSpikes and ProbSpikes
                "start": args.tsr_start,
                "stop": args.tsr_stop,
            },
            "tgtSpikeCount": {True: args.sc_true, False: args.sc_false},
        }
    },
}

device = torch.device("cuda:2")
writer = SummaryWriter(".")
net = SlayerMM(params, args.output_size).to(device)

error = snn.loss(params).to(device)

if args.loss_type == 0:
    criteria = error.numSpikes
elif args.loss_type == 1:
    criteria = error.weightedNumSpikes

optimizer = torch.optim.RMSprop(
    net.parameters(), lr=args.lr, weight_decay=0.5
)

train_dataset = ViTacMMDataset(
    path=args.data_dir, sample_file=f"train_80_20_{args.sample_file}.txt", output_size=args.output_size
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)
test_dataset = ViTacMMDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=args.output_size
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)

def _train():
    correct = 0
    num_samples = 0
    losses = [0, 0, 0]
    net.train()
    for i, (tact, vis, target, label) in enumerate(train_loader):
        tact = tact.to(device)
        vis = vis.to(device)
        target = target.to(device)
        output = net.forward(tact, vis)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        num_samples += len(label)

        #spike_loss = error.weightedNumSpikes(output, target) # numSpikes
        spike_loss = criteria(output, target) # numSpikes
        
        loss = spike_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    writer.add_scalar("loss/train", spike_loss / len(train_loader), epoch)
    writer.add_scalar("acc/train", correct / num_samples, epoch)

    return spike_loss

def _test():
    correct = 0
    num_samples = 0
    losses = [0, 0, 0]
    net.eval()
    with torch.no_grad():
        for i, (tact, vis, target, label) in enumerate(test_loader):
            tact = tact.to(device)
            vis = vis.to(device)
            target = target.to(device)
            output = net.forward(tact, vis)
            correct += torch.sum(snn.predict.getClass(output) == label).data.item()
            num_samples += len(label)

            #spike_loss = error.weightedNumSpikes(output, target) # numSpikes
            spike_loss = criteria(output, target) # numSpikes


        writer.add_scalar("loss/test", spike_loss / len(test_loader), epoch)
        writer.add_scalar("acc/test", correct / num_samples, epoch)

    return spike_loss


def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"mm_weights_{epoch:03d}_{args.sample_file:02d}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)

for epoch in range(1, args.epochs + 1):
    _train()
    if epoch % 10 == 0:
        test_loss = _test()
    if epoch % 50 == 0:
        _save_model(epoch)
