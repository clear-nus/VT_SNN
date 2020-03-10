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
pooled_viz_dir = args.data_dir + "pooled_vis/"


#sample_file = 1
trainingSet = ViTacDataset(datasetPath = pooled_viz_dir, sampleFile = args.data_dir + "/train_" + split_list[args.sample_file-1] + ".txt")
trainLoader = DataLoader(dataset=trainingSet, args.batch_size=8, shuffle=False, num_workers=8)
 
testingSet = ViTacDataset(datasetPath = pooled_viz_dir, sampleFile  = args.data_dir + "/test_" + split_list[args.sample_file-1] + ".txt")
testLoader = DataLoader(dataset=testingSet, args.batch_size=8, shuffle=False, num_workers=8)



class MultiMLP_LSTM(nn.Module):

    def __init__(self):
        super(MultiMLP_LSTM, self).__init__()
        self.input_size = 1000
        self.hidden_dim = args.hidden_size #30
        self.batch_size = 8
        self.num_layers = 1

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_size, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, 20)
        
        #self.pool_vis = nn.AvgPool3d((1,5,5), padding=0, stride=(1,7,7))
        self.fc_vis = nn.Linear(63*50*2, self.input_size)


    def forward(self, in_vis):

        in_vis = in_vis.reshape([in_vis.shape[0], 325, 50*63*2])
        #print('in vis:', in_vis.shape)
        embeddings = self.fc_vis(in_vis).permute(1,0,2)
        #print('embeddings:', embeddings.shape)
                
        # GRU input type: (seq_len, batch, input_size)
        out, hidden = self.gru(embeddings)
        out = out.permute(1,0,2)
        #print('out: ', out.shape)
        
        
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


# In[75]:


net = MultiMLP_LSTM().to(device)
# Create snn loss instance.
criterion = nn.CrossEntropyLoss()
# Define optimizer module.
optimizer = torch.optim.RMSprop(net.parameters(), lr = args.lr)


# In[76]:


for epoch in range(1, agrs.epoch+1):
    # Training loop.
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (in_vis, label) in enumerate(trainLoader, 0):

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
    train_acc = correct/len(trainLoader.dataset)
    writer.add_scalar("loss/train", batch_loss/len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)

    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_vis, label) in enumerate(testLoader, 0):
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

    test_acc = correct/len(testLoader.dataset)
    writer.add_scalar("loss/test", batch_loss/len(test_loader.dataset), epoch)
    writer.add_scalar("acc/test", test_acc, epoch)

    if epoch%100 == 0:
        _save_model(epoch, batch_loss/len(test_loader.dataset))



# In[ ]:




