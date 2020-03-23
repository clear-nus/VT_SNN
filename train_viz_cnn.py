#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from dataset import ViTacVisDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch import nn
import torch.nn.functional as F



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
    "--batch_size", type=int, help="Batch Size.", required=True
)
parser.add_argument(
    "--output_size", type=int, help="Batch Size.", required=True
)

args = parser.parse_args()


# class FLAGS():
#     def __init__(self):
#         self.data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN/'
#         self.batch_size = 8
#         self.sample_file = 1
#         self.lr = 0.00001
#         self.epochs = 2000
#         self.output_size = 20
# args = FLAGS()


device = torch.device("cuda:2")
writer = SummaryWriter(".")



train_dataset = ViTacVisDataset(
    path=args.data_dir, sample_file=f"train_80_20_{args.sample_file}.txt", output_size=args.output_size
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)
test_dataset = ViTacVisDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=args.output_size
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)



class CNN3D(nn.Module):

    def __init__(self):
        
        super(CNN3D, self).__init__()
        self.input_size = 8*5*3
        self.hidden_dim = 32
        self.num_layers = 1
        
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(10,5,5), stride=(5,2,2))
        self.conv2 = nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(5,3,3), stride=(3,2,2))
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5,3,3), stride=(3,2,2))

        # Define the output layer
        self.fc = nn.Linear(np.prod([8, 6, 6, 5]), 20)
        
        self.drop = nn.Dropout(0.5)
        
        #self.fc_mlp = nn.Linear(6300, self.input_size)

    def forward(self, x):
        
        #print('Model input ', x.size())
        batch_size, C, H, W, sequence_size = x.size()
        x = x.view([batch_size, C, sequence_size, H, W])
        #print('Model input ', x.size())
        
        # pass to cnn3d
        out = self.conv1(x)
        out = F.relu(out)
        #print('conv1 out: ', out.shape)
        out = self.conv2(out)
        out = F.relu(out)
        #print('conv2 out: ', out.shape)
        out = self.conv3(out)
        out = F.relu(out)
        #print('conv3 out: ', out.shape)
        out = out.view([batch_size, np.prod([8, 6, 6, 5])])
        out = self.fc(out)
        # use dropout to have better generalization
        out = self.drop(out)
        return out

def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)


net = CNN3D().to(device)
# Create snn loss instance.
criterion = nn.CrossEntropyLoss()
# Define optimizer module.
optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)



for epoch in range(1, args.epochs+1):
    # Training loop.
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (in_viz, _, label) in enumerate(train_loader, 0):

        in_viz = in_viz.to(device)
        label = label.to(device)
        # Forward pass of the network.
        #print(in_viz.shape)
        out = net.forward(in_viz)
        #print(out_tact.shape)
        # Calculate loss.
        #print(label.shape)
        loss = criterion(out, label)
        #print(loss)

        batch_loss += loss.cpu().data.item()
        # Reset gradients to zero.
        optimizer.zero_grad()
        # Backward pass of the network.
        loss.backward()
        # Update weights.
        optimizer.step()

        _, predicted = torch.max(out.data, 1)
        correct += (predicted == label).sum().item()

    # Reset training stats.
    #train_acc = correct/len(train_loader.dataset)

    #print(train_acc, batch_loss)

    # training res without dropout
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_viz, _, label) in enumerate(train_loader, 0):
            in_viz = in_viz.to(device)
            # Forward pass of the network.
            out = net.forward(in_viz)
            label = label.to(device)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == label).sum().item()
            # Calculate loss.
            loss = criterion(out, label)
            batch_loss += loss.cpu().data.item()

    train_acc = correct/len(train_loader.dataset)
    writer.add_scalar("loss/train", batch_loss/len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)

    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_viz, _, label) in enumerate(test_loader, 0):
            in_viz = in_viz.to(device)
            # Forward pass of the network.
            out = net.forward(in_viz)
            label = label.to(device)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == label).sum().item()
            # Calculate loss.
            loss = criterion(out, label)
            batch_loss += loss.cpu().data.item()

    test_acc = correct/len(test_loader.dataset)
    writer.add_scalar("loss/test", batch_loss/len(test_loader.dataset), epoch)
    writer.add_scalar("acc/test", test_acc, epoch)
    
    if epoch%100 == 0:
        _save_model(epoch, batch_loss/len(test_loader.dataset))