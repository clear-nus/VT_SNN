#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import slayerSNN as snn
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()

parser = argparse.ArgumentParser("Train model.")

parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file", type=int, help="Sample number to train from.", required=True
)

parser.add_argument(
    "--batch_size", type=int, help="Batch Size.", required=True
)
parser.add_argument(
    "--cuda", type=int, help="Cuda Number.", required=True
)

args = parser.parse_args()


# lr = 0.01
# batch_size=8
# sample_file=2

output_size = 20
input_size = 156
hidden_size = 32
data_dir='/home/tasbolat/some_python_examples/data_VT_SNN/'

epochs = 2000

if args.batch_size >= 200:
    epochs = 5000

save_dir = '/home/tasbolat/some_python_examples/VT_SNN/analysis_of_models/results/'

class ViTacDataset(Dataset):
    def __init__(self, path, sample_file, output_size):
        self.path = path
        self.output_size = output_size
        sample_file = Path(path) / sample_file
        self.samples = np.loadtxt(sample_file).astype("int")
        tact = torch.load(Path(path) / "tact.pt")
        self.tact = tact.reshape(tact.shape[0], -1, 1, 1, tact.shape[-1])

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1
    
        return (
            self.tact[input_index],
            torch.tensor(0),
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]
params = {
    "neuron": {
        "type": "SRMALPHA",
        "theta": 10,
        "tauSr": 10.0,
        "tauRef": 1.0,
        "scaleRef": 2,
        "tauRho": 1,
        "scaleRho": 1,
    },
    "simulation": {"Ts": 1.0, "tSample": 325, "nSample": 1},
    "training": {
        "error": {
            "type": "NumSpikes",  # "NumSpikes" or "ProbSpikes"
            "probSlidingWin": 20,  # only valid for ProbSpikes
            "tgtSpikeRegion": {  # valid for NumSpikes and ProbSpikes
                "start": 0,
                "stop": 325,
            },
            "tgtSpikeCount": {True: 150, False: 5},
        }
    },
}


class SlayerMLP(torch.nn.Module):
    '''
    2 layer MLP based on SLAYER used for tactile data
    '''
    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerMLP, self).__init__()
        self.output_size = output_size
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))
        self.spike_trains = [spike_1]
        return spike_output

class SlayerVisMLP(torch.nn.Module):
    def __init__(self, netParams, hidden_size, output_size):
        super(SlayerVisMLP, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((50, 63, 2), hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, downsampled):
        spike_1 = self.slayer.spike(self.slayer.psp(downsampled)) # 32, 32, 16
        spike_2 = self.slayer.spike(self.fc1(self.slayer.psp(spike_1))) #  10
        spike_3 = self.slayer.spike(self.fc2(self.slayer.psp(spike_2)))
        self.spike_trains = [spike_1, spike_2]
        return spike_3

device = torch.device("cuda:"+str(args.cuda))



train_dataset = ViTacDataset(
    path=data_dir, sample_file=f"train_80_20_{args.sample_file}.txt", output_size=output_size
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_dataset = ViTacDataset(
    path=data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=output_size
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)

net = SlayerMLP(params, input_size, hidden_size, output_size).to(device)
error = snn.loss(params).to(device)
optimizer = torch.optim.RMSprop(
    net.parameters(), lr=args.lr, weight_decay=0.5
)

tr_losses=[]
te_losses=[]
tr_accs=[]
te_accs=[]

for epoch in range(1, epochs+1):
    correct = 0
    num_samples = 0
    batch_loss = 0
    net.train()
    for i, (tact, _, target, label) in enumerate(train_loader):
        
        tact = tact.to(device)
        
        target = target.to(device)
        output = net.forward(tact)
        
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        loss = error.numSpikes(output, target)
        batch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    tr_losses.append(batch_loss / len(train_loader.dataset) )
    tr_accs.append(correct / len(train_loader.dataset) )
    
    correct = 0
    num_samples = 0
    batch_loss = 0
    net.eval()
    with torch.no_grad():
        for i, (tact, _, target, label) in enumerate(test_loader):

            tact = tact.to(device)

            target = target.to(device)
            output = net.forward(tact)

            correct += torch.sum(snn.predict.getClass(output) == label).data.item()
            loss = error.numSpikes(output, target)
            batch_loss += loss.item()

    te_losses.append(batch_loss / len(test_loader.dataset) )
    te_accs.append(correct / len(test_loader.dataset) )
    
    if epoch%50 == 0:
        print('Epoch ', epoch, '---------------------------------')
        print('Train, Test acc: ', tr_accs[-1], te_accs[-1])



torch.save(net.state_dict(), save_dir + 'models/tact_' + str(args.batch_size) + '_' + str(args.lr) + '.pt')


import pickle
pickle.dump([tr_losses, te_losses, tr_accs, te_accs], open(save_dir + 'stats/tact' + str(args.batch_size) + '_' + str(args.lr) + '.pk', 'wb'))

fig, ax = plt.subplots(2, figsize=(15,7))
ax[0].set_title('batch size = ' + str(args.batch_size) + ', lr = ' + str(args.lr))
ax[0].plot(tr_losses)
ax[0].plot(te_losses)
ax[0].legend(['Train', 'Test'])
ax[0].set_ylabel('Loss')

ax[1].plot(tr_accs)
ax[1].plot(te_accs)
ax[1].legend(['Train', 'Test'])
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epochs')
plt.tight_layout()
fig.savefig(save_dir + 'visual/tact_' + str(args.batch_size) + '_' + str(args.lr) + '.png')




