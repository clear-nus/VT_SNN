#!/usr/bin/env python
# coding: utf-8

import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../../slayerPytorch/src/")
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
from IPython.display import HTML
import zipfile
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch
import numpy as np
import copy


# In[2]:


netParams = snn.params('../tact_learning/network.yaml')

netParams['simulation']['tSample'] = 65
netParams['training']['error']['tgtSpikeRegion']['stop'] = 65
netParams['training']['error']['tgtSpikeCount'][True] = 50 # 30 # 15
netParams['training']['error']['tgtSpikeCount'][False] = 3 #5 # 13


ref_name = 'models_and_stats2/mlp_both_0'
device = torch.device('cuda:0')


# Dataset definition
class ViTacDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, modals=0):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.modality = modals

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        desiredClass = torch.zeros((20, 1, 1, 1))
        desiredClass[classLabel,...] = 1

        if self.modality == 0:
            inputSpikes_tact = np.load(self.path + str(inputIndex.item()) + '_tact.npy')
            #inputSpikes_tact = np.delete(inputSpikes_tact, [], 0)
            inputSpikes_tact = torch.FloatTensor(inputSpikes_tact)#[39:, :,:]
            return inputSpikes_tact.reshape((-1, 1, 1, inputSpikes_tact.shape[-1])), desiredClass, classLabel
        elif self.modality == 1:
            inputSpikes_vis = np.load(self.path + str(inputIndex.item()) + '_vis.npy')
            inputSpikes_vis = torch.FloatTensor(inputSpikes_vis)
            return inputSpikes_vis, desiredClass, classLabel
        elif self.modality == 2:
            inputSpikes_tact = np.load(self.path + str(inputIndex.item()) + '_tact.npy')
            inputSpikes_tact = torch.FloatTensor(inputSpikes_tact)
            inputSpikes_vis = np.load(self.path + str(inputIndex.item()) + '_vis.npy')
            inputSpikes_vis = torch.FloatTensor(inputSpikes_vis)
            return inputSpikes_tact.reshape((-1, 1, 1, inputSpikes_tact.shape[-1])), inputSpikes_vis, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]

# Dataset and dataLoader instances.
split_list = ['80_20_1','80_20_2','80_20_3','80_20_4','80_20_5']

training_loader = []
testing_loader = []

for k in range(5):
    
    trainingSet = ViTacDataset(datasetPath = "../bd_data/bd_001_grasp_lift_hold/", 
                                sampleFile = "../bd_data/bd_data_001s_new_splits/train_" + split_list[k] + ".txt",
                               modals=2)
    trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=8)
    training_loader.append(trainLoader)
    
    testingSet = ViTacDataset(datasetPath = "../bd_data/bd_001_grasp_lift_hold/", 
                                sampleFile  = "../bd_data/bd_data_001s_new_splits/test_" + split_list[k] + ".txt", 
                              modals=2)
    testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=8)
    testing_loader.append(testLoader)
in1, in2, _, label  = trainingSet[0]
in1.shape, in2.shape


# In[6]:


class EncoderVis(torch.nn.Module):
    def __init__(self, netParams):
        super(EncoderVis, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.pool1 = slayer.pool(5, stride=7)
        self.fc1   = slayer.dense((43, 50, 2), 10)

    def forward(self, spikeInput):
        # Both set of definitions are equivalent. The uncommented version is much faster.
        spikeLayer1 = self.slayer.spike(self.pool1(self.slayer.psp(spikeInput ))) # 32, 32, 16
        spikeLayer5 = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer1))) #  10
        return spikeLayer5
    
class EncoderTact(torch.nn.Module):
    def __init__(self, netParams):
        super(EncoderTact, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.fc1   = slayer.dense((39*2*2), 50)

    def forward(self, spikeInput):        
        spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp((spikeInput))))
        
        return spikeLayer1


# In[7]:


class MultiModalSystem(torch.nn.Module):
    def __init__(self, netParams):
        super(MultiModalSystem, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.tactile = EncoderTact(netParams)
        self.vision = EncoderVis(netParams)
        self.slayer = slayer
        self.fc1   = slayer.dense(60, 20)

    def forward(self, spikeInputTact, spikeInputVis):
        spikeLayer1 = self.tactile(spikeInputTact)
        spikeLayer2 = self.vision(spikeInputVis)
        spikeAll = torch.cat([spikeLayer1, spikeLayer2], dim=1)
        #print(spikeLayer1.shape, spikeLayer2.shape, spikeAll.shape)
        out = self.slayer.spike(self.slayer.psp(self.fc1(spikeAll)))
        return out


# In[8]:




# In[9]:


train_accs = {0:[],1:[],2:[],3:[],4:[]}
test_accs = {0:[],1:[],2:[],3:[],4:[]}
train_loss = {0:[],1:[],2:[],3:[],4:[]}
test_loss = {0:[],1:[],2:[],3:[],4:[]}


# In[ ]:


for k in range(5):
    print(k, 'split --------------------------------------------------')
    trainLoader = training_loader[k]
    testLoader = testing_loader[k]
    net = MultiModalSystem(netParams).to(device)
    error = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
    for epoch in range(501):
        tSt = datetime.now()
        # Training loop.
        net.train()
        correct = 0
        batch_loss = 0
        train_acc = 0
        for i, (input_tac, input_vis, target, label) in enumerate(trainLoader, 0):

            input_vis = input_vis.to(device)
            input_tac = input_tac.to(device)
            target = target.to(device)
            # Forward pass of the network.
            out_tact = net.forward(input_tac, input_vis)
            # Calculate loss.
            loss = error.numSpikes(out_tact, target)
            l2_regularization = torch.FloatTensor([0]).to(device)
            for param in net.parameters():
                l2_regularization += torch.norm(param, 2)
            loss += 0.001*l2_regularization[0]
            batch_loss += loss.cpu().data.item()
            # Reset gradients to zero.
            optimizer.zero_grad()
            # Backward pass of the network.
            loss.backward()
            # Update weights.
            optimizer.step()
            correct += torch.sum( snn.predict.getClass(out_tact) == label ).data.item()

        # Reset training stats.
        train_acc = correct/len(trainLoader.dataset)
        train_loss[k].append(batch_loss)
        train_accs[k].append(train_acc)

        # testing
        net.eval()
        correct = 0
        batch_loss = 0
        test_acc = 0
        with torch.no_grad():
            for i, (input_tac, input_vis, target, label) in enumerate(testLoader, 0):
                input_vis = input_vis.to(device)
                input_tac = input_tac.to(device)
                target = target.to(device)
                # Forward pass of the network.
                out_tact = net.forward(input_tac, input_vis)

                correct += torch.sum( snn.predict.getClass(out_tact) == label ).data.item()
                # Calculate loss.
                loss = error.numSpikes(out_tact, target)
                batch_loss += loss.cpu().data.item()

        test_loss[k].append(batch_loss)
        test_acc = correct/len(testLoader.dataset)
        test_accs[k].append(test_acc)
        if epoch%20 == 0:
            print('------------------------')
            print('saving model')
            torch.save(net.state_dict(), ref_name + '_' + str(epoch) + '_' + str(k) + ".pt")
            print('Train:', train_acc, 'Test:', test_acc)
            print('------------------------')
        if epoch%50 == 0:
            print(epoch, 'Train:', train_acc, 'Test:', test_acc)
    del net
    
    
import pickle
pickle.dump( [train_accs, test_accs, train_loss, test_loss,], open( ref_name + ".stats", "wb" ) )
