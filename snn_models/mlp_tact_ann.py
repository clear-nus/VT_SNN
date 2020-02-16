#!/usr/bin/env python
# coding: utf-8

# In[1]:



from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from learningStats import learningStats
import zipfile
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
import numpy as np
import copy


ref_name = 'models_and_stats2/mlp_gru_tact_0'
device = torch.device('cuda:0')
num_epochs = 2



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
                               modals=0)
    trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=8)
    training_loader.append(trainLoader)
    
    testingSet = ViTacDataset(datasetPath = "../bd_data/bd_001_grasp_lift_hold/", 
                                sampleFile  = "../bd_data/bd_data_001s_new_splits/test_" + split_list[k] + ".txt", 
                              modals=0)
    testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=8)
    testing_loader.append(testLoader)


class MLP_LSTM(nn.Module):

    def __init__(self):
        super(MLP_LSTM, self).__init__()
        self.seq_length = 65
        self.hidden_dim = 30
        self.batch_size = 8
        self.num_layers = 1

        # Define the LSTM layer
        self.lstm = nn.GRU(self.seq_length, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, 20)
        
        self.fc_mlp = nn.Linear(156, 50)

    def init_hidden(self, bs):
        # This is what we'll initialise our hidden state as
        self.batch_size = bs
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_data):

        lstm_in = self.fc_mlp(input_data).permute(0,2,1)
        #print('mlp:', lstm_in.shape, len(lstm_in))
        
        lstm_out, self.hidden = self.lstm(lstm_in)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.fc(lstm_out[:, -1, :])
        
        #print(y_pred.shape)
        return y_pred




# In[7]:


train_accs = {0:[],1:[],2:[],3:[],4:[]}
test_accs = {0:[],1:[],2:[],3:[],4:[]}
train_loss = {0:[],1:[],2:[],3:[],4:[]}
test_loss = {0:[],1:[],2:[],3:[],4:[]}


# In[8]:


for k in range(5):
    print(k, ' split is running ............')
    trainLoader = training_loader[k]
    testLoader = testing_loader[k]
    # Define model
    net = MLP_LSTM().to(device)
    # Create snn loss instance.
    criterion = nn.CrossEntropyLoss()
    # Define optimizer module.
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
    for epoch in range(501):
        tSt = datetime.now()
        # Training loop.
        net.train()
        correct = 0
        batch_loss = 0
        train_acc = 0
        for i, (input_tact_left, _, label) in enumerate(trainLoader, 0):

            input_tact_left = input_tact_left.to(device)
            input_tact_left = input_tact_left.squeeze()
            input_tact_left = input_tact_left.permute(0,2,1)
            #print(input_tact_left.shape)
            label = label.to(device)
            # Forward pass of the network.
            net.hidden = net.init_hidden(input_tact_left.shape[0])
            out_tact = net.forward(input_tact_left)
            #print(out_tact.shape)
            # Calculate loss.
            #print(label.shape)
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
        train_loss[k].append(batch_loss)
        train_accs[k].append(train_acc)
        #print(train_acc, batch_loss)

        # testing
        net.eval()
        correct = 0
        batch_loss = 0
        test_acc = 0
        with torch.no_grad():
            for i, (input_tact_left, _, label) in enumerate(testLoader, 0):
                input_tact_left = input_tact_left.to(device)
                input_tact_left = input_tact_left.squeeze()
                input_tact_left = input_tact_left.permute(0,2,1)

                # Forward pass of the network.
                net.hidden = net.init_hidden(input_tact_left.shape[0])
                out_tact = net.forward(input_tact_left)
                label = label.to(device)
                _, predicted = torch.max(out_tact.data, 1)
                correct += (predicted == label).sum().item()
                # Calculate loss.
                loss = criterion(out_tact, label)
                batch_loss += loss.cpu().data.item()

        test_loss[k].append(batch_loss)
        test_acc = correct/len(testLoader.dataset)
        test_accs[k].append(test_acc)
        if epoch%20==0:
            print('------------------------')
            print('saving model')
            torch.save(net.state_dict(), ref_name + '_' + str(epoch) + '_' + str(k) + ".pt")
            print('Train:', train_acc, 'Test:', test_acc)
            print('------------------------')
        if epoch%50 == 0:
            print(epoch, 'Train:', train_acc, 'Test:', test_acc)

import pickle
pickle.dump( [train_accs, test_accs, train_loss, test_loss], open( ref_name + ".stats", "wb" ) )



