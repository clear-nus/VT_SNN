#!/usr/bin/env python
# coding: utf-8


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


# class FLAGS():
#     def __init__(self):
#         self.data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN/'
#         self.batch_size = 8
#         self.sample_file = 1
#         self.lr = 0.0001
#         self.epochs = 400
# args = FLAGS()

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
parser.add_argument(
    "--batch_size", type=int, help="Batch Size.", required=True
)
parser.add_argument(
    "--output_size", type=int, help="Number of classes.", required=True
)

args = parser.parse_args()

# Dataset definition
class ViTacDataset(Dataset):
    def __init__(self, datasetPath, sampleFile):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.vis = torch.load(Path(self.path)  / "ds_vis.pt")
        tact = torch.load(Path(self.path) / "tact.pt")
        self.tact = tact.reshape(tact.shape[0], -1, tact.shape[-1])

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        return self.tact[inputIndex], self.vis[inputIndex], classLabel

    def __len__(self):
        return self.samples.shape[0]

# Dataset and dataLoader instances.
split_list = ['80_20_1','80_20_2','80_20_3','80_20_4','80_20_5']

trainingSet = ViTacDataset(datasetPath = args.data_dir, sampleFile = args.data_dir + "/train_" + split_list[args.sample_file-1] + ".txt")
train_loader = DataLoader(dataset=trainingSet, batch_size=args.batch_size, shuffle=False, num_workers=8)
 
testingSet = ViTacDataset(datasetPath = args.data_dir, sampleFile  = args.data_dir + "/test_" + split_list[args.sample_file-1] + ".txt")
test_loader = DataLoader(dataset=testingSet, batch_size=args.batch_size, shuffle=False, num_workers=8)

class MultiMLP_LSTM(nn.Module):
    def __init__(self):
        super(MultiMLP_LSTM, self).__init__()
        self.input_size = 150+78*2
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, args.output_size)
        
        #self.pool_vis = nn.AvgPool3d((1,5,5), padding=0, stride=(1,7,7))
        self.fc_vis = nn.Linear(63*50*2, self.input_size-78*2)


    def forward(self, in_tact, in_vis):
        in_vis = in_vis.reshape([in_vis.shape[0],  in_vis.shape[-1], 50*63*2])
        viz_embeddings = self.fc_vis(in_vis).permute(1,0,2)
        in_tact = in_tact.permute(2,0,1)
        embeddings = torch.cat([viz_embeddings, in_tact], dim=2)
        out, hidden = self.gru(embeddings)
        out = out.permute(1,0,2)
        y_pred = self.fc(out[:, -1, :])
        
        return y_pred

device = torch.device("cuda")
writer = SummaryWriter(".")

def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)

net = MultiMLP_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr = args.lr)

for epoch in range(1, args.epochs+1):
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (in_tac, in_vis, label) in enumerate(train_loader, 0):

        in_vis = in_vis.to(device)
        in_tac = in_tac.to(device)
        label = label.to(device)
        out_tact = net.forward(in_tac, in_vis)
        loss = criterion(out_tact, label)

        batch_loss += loss.cpu().data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out_tact.data, 1)
        correct += (predicted == label).sum().item()

    # Reset training stats.
    train_acc = correct/len(train_loader.dataset)
    writer.add_scalar("loss/train", batch_loss/len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)
    #print(train_acc, batch_loss)

    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_tac, in_vis, label) in enumerate(test_loader, 0):
            in_vis = in_vis.to(device)
            in_tac = in_tac.to(device)

            # Forward pass of the network.
            out_tact = net.forward(in_tac, in_vis)
            label = label.to(device)
            _, predicted = torch.max(out_tact.data, 1)
            correct += (predicted == label).sum().item()
            # Calculate loss.
            loss = criterion(out_tact, label)
            batch_loss += loss.cpu().data.item()

    test_acc = correct/len(test_loader.dataset)
    writer.add_scalar("loss/test", batch_loss/len(test_loader.dataset), epoch)
    writer.add_scalar("acc/test", test_acc, epoch)

    if epoch%100 == 0:
        _save_model(epoch, batch_loss/len(test_loader.dataset))
