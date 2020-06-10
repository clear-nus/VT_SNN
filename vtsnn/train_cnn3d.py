"""Train the CNN3D models."""

from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
import copy
from pathlib import Path
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from vtsnn.dataset import ViTacDataset
from vtsnn.models.cnn3d import TactCNN3D, VisCNN3D, MmCNN3D

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


parser = argparse.ArgumentParser("Train MLP-GRU model.")
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
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["tact", "vis", "mm"],
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

args = parser.parse_args()

if args.task == "cw":
    output_size = 20
else:  # Slip
    output_size = 2

if args.mode == "tact":
    model = TactCNN3D
elif args.mode == "vis":
    model = VisCNN3D
else:
    model = MmCNN3D

train_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"train_80_20_{args.sample_file}.txt",
    output_size=output_size,
    mode=args.mode,
    spiking=False,
    rectangular=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)

test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=output_size,
    mode=args.mode,
    spiking=False,
    rectangular=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)


device = torch.device("cuda")
writer = SummaryWriter(".")

net = model(output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)


def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(args.checkpoint_dir) / f"weights-{epoch:03d}.pt"
    torch.save(net.state_dict(), checkpoint_path)


def _train(epoch):
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for *inputs, _, label in train_loader:
        inputs = [i.to(device) for i in inputs]
        label = label.to(device)
        output = net.forward(*inputs)
        loss = criterion(output, label)

        batch_loss += loss.cpu().data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum().item()

    train_acc = correct / len(train_loader.dataset)
    writer.add_scalar(
        "loss/train", batch_loss / len(train_loader.dataset), epoch
    )
    writer.add_scalar("acc/train", train_acc, epoch)


def _test(epoch):
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for *inputs, _, label in test_loader:
            inputs = [i.to(device) for i in inputs]
            label = label.to(device)
            output = net.forward(*inputs)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            loss = criterion(output, label)
            batch_loss += loss.cpu().data.item()

    test_acc = correct / len(test_loader.dataset)
    writer.add_scalar("loss/test", batch_loss / len(test_loader.dataset), epoch)
    writer.add_scalar("acc/test", test_acc, epoch)


for epoch in range(1, args.epochs + 1):
    _train(epoch)
    if epoch % 50 == 0:
        _test(epoch)
    if epoch % 100 == 0:
        _save_model(epoch)
