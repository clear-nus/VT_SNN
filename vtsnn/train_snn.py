from pathlib import Path
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn

from vtsnn.models.snn import SlayerMLP, SlayerMM
from vtsnn.dataset import ViTacDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

parser = argparse.ArgumentParser("Train VT-SNN models.")

parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)
parser.add_argument(
    "--network_config",
    type=str,
    help="Path SNN network configuration.",
    required=True,
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["tact", "vis", "vistact"],
    help="Type of model to run.",
    required=True,
)

parser.add_argument(
    "--task",
    type=str,
    help="The classification task.",
    choices=["cw", "slip"],
    required=True,
)

parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)

parser.add_argument(
    "--loss",
    type=str,
    help="Loss function to use.",
    choices=["NumSpikes", "WeightedNumSpikes"],
    required=True,
)
args = parser.parse_args()

LOSS_TYPES = ["NumSpikes", "WeightedNumSpikes"]

params = snn.params(args.network_config)

if args.task == "cw":
    output_size = 20
else:  # Slip
    output_size = 2

if args.mode == "tact":
    model = SlayerMLP
    model_args = {
        "params": params,
        "input_size": 156,
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
elif args.mode == "vis":
    model = SlayerMLP
    model_args = {
        "params": params,
        "input_size": (50, 63, 2),
        "hidden_size": args.hidden_size,
        "output_size": output_size,
    }
else:  # NOTE: args.hidden_size unused here
    model = SlayerMM
    model_args = {
        "params": params,
        "tact_input_size": 156,
        "vis_input_size": (50, 63, 2),
        "tact_output_size": 50,
        "vis_output_size": 10,
        "output_size": output_size,
    }

device = torch.device("cuda")
writer = SummaryWriter(".")
net = model(**model_args).to(device)


if args.loss == "NumSpikes":
    params["training"]["error"]["type"] = "NumSpikes"
    error = snn.loss(params).to(device)
    criteria = error.numSpikes
elif args.loss == "WeightedNumSpikes":
    params["training"]["error"]["type"] = "WeightedNumSpikes"
    error = snn.loss(params).to(device)
    criteria = error.weightedNumSpikes

optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.5)

train_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"train_80_20_{args.sample_file}.txt",
    output_size=output_size,
    spiking=True,
    mode=args.mode,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
)
test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=output_size,
    spiking=True,
    mode=args.mode,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)

def _train():
    correct = 0
    num_samples = 0
    net.train()
    for *data, target, label in train_loader:
        data = [d.to(device) for d in data]
        target = target.to(device)
        output = net.forward(*data)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        num_samples += len(label)
        spike_loss = criteria(output, target)

        optimizer.zero_grad()
        spike_loss.backward()
        optimizer.step()

    writer.add_scalar("loss/train", spike_loss / len(train_loader), epoch)
    writer.add_scalar("acc/train", correct / num_samples, epoch)


def _test():
    correct = 0
    num_samples = 0
    net.eval()
    with torch.no_grad():
        for *data, target, label in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = net.forward(*data)
            correct += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            num_samples += len(label)
            spike_loss = criteria(output, target)  # numSpikes

        writer.add_scalar("loss/test", spike_loss / len(test_loader), epoch)
        writer.add_scalar("acc/test", correct / num_samples, epoch)


def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(args.checkpoint_dir) / f"weights_{epoch:03d}.pt"
    torch.save(net.state_dict(), checkpoint_path)


for epoch in range(1, args.epochs + 1):
    _train()
    if epoch % 10 == 0:
        _test()
    if epoch % 100 == 0:
        _save_model(epoch)
