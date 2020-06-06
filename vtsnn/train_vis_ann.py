from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import zipfile
import torch
from torch import nn
import numpy as np
import copy
from pathlib import Path
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from vtsnn.dataset import ViTacVisDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


parser = argparse.ArgumentParser("Train model.")
parser.add_argument("--epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir", type=str, help="Path for saving checkpoints.", required=True
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file", type=int, help="Sample number to train from.", required=True
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)
parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
parser.add_argument("--output_size", type=int, help="Number of classes.", required=True)

args = parser.parse_args()

train_dataset = ViTacVisDataset(
    path=args.data_dir,
    sample_file=f"train_80_20_{args.sample_file}.txt",
    output_size=args.output_size,
    spike=False,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
)

test_dataset = ViTacVisDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=args.output_size,
    spike=False,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
)


class Vis_MLP_GRU(nn.Module):
    def __init__(self):
        super(Vis_MLP_GRU, self).__init__()
        self.input_size = 1000
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, args.output_size)

        # self.pool_vis = nn.AvgPool3d((1,5,5), padding=0, stride=(1,7,7))
        self.fc_vis = nn.Linear(63 * 50 * 2, self.input_size)

    def forward(self, in_vis):
        in_vis = in_vis.reshape([in_vis.shape[0], in_vis.shape[-1], 50 * 63 * 2])
        embeddings = self.fc_vis(in_vis).permute(1, 0, 2)
        out, hidden = self.gru(embeddings)
        out = out.permute(1, 0, 2)

        y_pred = self.fc(out[:, -1, :])

        return y_pred


device = torch.device("cuda")
writer = SummaryWriter(".")

net = Vis_MLP_GRU().to(device)
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
    for i, (in_vis, _, label) in enumerate(train_loader):
        in_vis = in_vis.to(device)
        label = label.to(device)
        out_vis = net.forward(in_vis)
        loss = criterion(out_vis, label)

        batch_loss += loss.cpu().data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out_vis.data, 1)
        correct += (predicted == label).sum().item()

    train_acc = correct / len(train_loader.dataset)
    writer.add_scalar("loss/train", batch_loss / len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)


def _test(epoch):
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_vis, _, label) in enumerate(test_loader, 0):
            in_vis = in_vis.to(device)
            out_vis = net.forward(in_vis)
            label = label.to(device)
            _, predicted = torch.max(out_vis.data, 1)
            correct += (predicted == label).sum().item()
            loss = criterion(out_vis, label)
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
