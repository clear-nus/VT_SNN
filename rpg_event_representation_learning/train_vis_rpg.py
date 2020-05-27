#!/usr/bin/env python
# coding: utf-8

# # EST
# 


import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from IPython.display import HTML
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch
import numpy as np
import torch.nn as nn
from utils.loader2 import Loader
from utils.models import Classifier
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


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

args = parser.parse_args()

# Dataset definition
class RawDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, modals=0):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile, dtype=str)
        

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = int(self.samples[index, 1])
        #desiredClass = torch.zeros((20, 1, 1, 1))
        #desiredClass[classLabel,...] = 1
        inputSpikes = torch.FloatTensor( np.load(self.path + inputIndex + '_vis.npy') )
        return inputSpikes, classLabel
        
    def __len__(self):
        return self.samples.shape[0]


# In[15]:


device = torch.device('cuda:2')
writer = SummaryWriter(".")


def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)

# class FLAGS3():
#     def __init__(self):
#         self.batch_size = 4
#         self.data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN/'
#         self.sample_file = 1
#         self.lr = 1e-4
#         self.epochs = 40
# args = FLAGS3()

class FLAGS():
    def __init__(self):
        self.batch_size = args.batch_size
        self.pin_memory =True
        self.num_workers = 1
        self.device = device
flags = FLAGS()


# In[17]:


# Dataset and dataLoader instances.
#split_list = ['80_20_1','80_20_2','80_20_3','80_20_4','80_20_5']

    
trainingSet = RawDataset(datasetPath = args.data_dir + 'vis_rpg_data/', 
                        sampleFile = args.data_dir + "/train_80_20_" + str(args.sample_file) + ".txt")
train_loader = Loader(trainingSet, flags, device=device)    

testingSet = RawDataset(datasetPath = args.data_dir + 'vis_rpg_data/', 
                        sampleFile  = args.data_dir + "/test_80_20_" + str(args.sample_file) + ".txt")
test_loader = Loader(testingSet, flags, device=device)
in1, _  = trainingSet[7]
in1.shape

# In[19]:


model = Classifier(num_classes=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)



criterion = torch.nn.CrossEntropyLoss()

def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)

for epoch in range(1, args.epochs+1):
    sum_loss = 0
    correct = 0
    model.train()
    for events, labels in train_loader:
        optimizer.zero_grad()
        out, vox = model.forward(events)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1)
        correct += (predicted == labels).sum().item()  
        sum_loss += loss.item()
    if epoch % 10 == 9:
        lr_scheduler.step()
    training_loss = sum_loss/ len(trainingSet)
    training_accuracy = correct / len(trainingSet)
    writer.add_scalar("loss/train", training_loss, epoch)
    writer.add_scalar("acc/train", training_accuracy, epoch)
    
    
    model.eval()
    correct = 0
    sum_loss = 0
    model.eval()
    with torch.no_grad():
        for events, labels in test_loader:       
            out, vox = model.forward(events)
            loss = criterion(out, labels)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == labels).sum().item()        
            sum_loss += loss.item()
    validation_loss = sum_loss / len(testingSet)
    validation_accuracy = correct / len(testingSet)
    writer.add_scalar("loss/test", validation_loss, epoch)
    writer.add_scalar("acc/test", validation_accuracy, epoch)

_save_model(args.epochs)