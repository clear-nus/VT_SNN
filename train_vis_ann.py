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
    "--output_size", type=int, help="Number of classes.", default=20
)

args = parser.parse_args()

# Dataset definition
class ViTacDataset(Dataset):
    def __init__(self, datasetPath, sampleFile):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.vis = torch.load(Path(self.path) / "ds_vis.pt")

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        return self.vis[inputIndex], classLabel

    def __len__(self):
        return self.samples.shape[0]


# In[4]:


# Dataset and dataLoader instances.
split_list = ['80_20_1','80_20_2','80_20_3','80_20_4','80_20_5']


#data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN/'


#sample_file = 1
trainingSet = ViTacDataset(datasetPath = args.data_dir, sampleFile = args.data_dir + "/train_" + split_list[args.sample_file-1] + ".txt")
train_loader = DataLoader(dataset=trainingSet, batch_size = args.batch_size, shuffle=False, num_workers=8)
 
testingSet = ViTacDataset(datasetPath = args.data_dir, sampleFile  = args.data_dir + "/test_" + split_list[args.sample_file-1] + ".txt")
test_loader = DataLoader(dataset=testingSet, batch_size = args.batch_size, shuffle=False, num_workers=8)



class Vis_MLP_GRU(nn.Module):

    def __init__(self):
        super(Vis_MLP_GRU, self).__init__()
        self.input_size = 1000
        self.hidden_dim = args.hidden_size #32
        self.batch_size = 8
        self.num_layers = 1

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_size, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, args.output_size)
        
        #self.pool_vis = nn.AvgPool3d((1,5,5), padding=0, stride=(1,7,7))
        self.fc_vis = nn.Linear(63*50*2, self.input_size)


    def forward(self, in_vis):
        in_vis = in_vis.reshape([in_vis.shape[0], in_vis.shape[-1], 50*63*2])
        #print('in vis:', in_vis.shape)
        embeddings = self.fc_vis(in_vis).permute(1,0,2)
        #print('embeddings:', embeddings.shape)
                
        # GRU input type: (seq_len, batch, input_size)
        out, hidden = self.gru(embeddings)
        out = out.permute(1,0,2)        
        
        # Only take the output from the final timetep
        #print('out:', out.shape)
        y_pred = self.fc(out[:, -1, :])
        
        return y_pred


# In[74]:

device = torch.device("cuda:2")
writer = SummaryWriter(".")


def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)



train_accs = []
test_accs = []
train_loss = []
test_loss = []


net = Vis_MLP_GRU().to(device)
# Create snn loss instance.
criterion = nn.CrossEntropyLoss()
# Define optimizer module.
optimizer = torch.optim.RMSprop(net.parameters(), lr = args.lr)


for epoch in range(1, args.epochs+1):
    # Training loop.
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (in_vis, label) in enumerate(train_loader, 0):

        in_vis = in_vis.to(device)
        #in_vis = in_vis.squeeze().permute(0,1,4,2,3)
        #print(in_vis.shape)
        label = label.to(device)
        # Forward pass of the network.
        out_tact = net.forward(in_vis)
        # Calculate loss.
        loss = criterion(out_tact, label)
        #print(loss)

        batch_loss += loss.cpu().data.item()
        # Reset gradients to zero.
        optimizer.zero_grad()
        # Backward pass of the network.
        loss.backward()
        # Update weights.
        optimizer.step()

        _, predicted = torch.max(out_tact.data, 1)
        correct += (predicted == label).sum().item()

    # Reset training stats.
    train_acc = correct/len(train_loader.dataset)
    writer.add_scalar("loss/train", batch_loss/len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)

    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_vis, label) in enumerate(test_loader, 0):
            in_vis = in_vis.to(device)
            #in_vis = in_vis.squeeze().permute(0,1,4,2,3)

            # Forward pass of the network.
            out_tact = net.forward(in_vis)
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
