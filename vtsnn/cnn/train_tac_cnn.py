import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from vtsnn.dataset import ViTacDataset


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
parser.add_argument(
    "--last_layer", type=int, help="last layer size for FC.", default=26
)

args = parser.parse_args()


device = torch.device("cuda:2")
writer = SummaryWriter(".")


train_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"train_80_20_{args.sample_file}.txt", output_size=args.output_size, rectangular=True
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=args.output_size, rectangular=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)



class CNN3D(nn.Module):

    def __init__(self):
        
        super(CNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(7,3,3), stride=(4,1,1))
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5,3,3), stride=(3,1,1))
        
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
        out = out.view([batch_size, np.prod([8, args.last_layer, 5, 3])])
        #print(out.shape)
        
        return out



class MyNet(nn.Module):

    def __init__(self):
        
        super(MyNet, self).__init__()
        
        self.cnn_left = CNN3D()
        self.cnn_right = CNN3D()

        # 8, 35, 5, 3
        # Define the output layer
        self.fc = nn.Linear(np.prod([8, args.last_layer, 5, 3])*2, args.output_size)
        
        self.drop = nn.Dropout(0.5)

    def forward(self, x_left, x_right):
        out_left = self.cnn_left(x_left)
        out_right = self.cnn_right(x_right)
        
        out = torch.cat([out_left, out_right], dim=1)
        
        out = self.fc(out)
        
        out = self.drop(out)
        return out



net = MyNet().to(device)
# Create snn loss instance.
criterion = nn.CrossEntropyLoss()
# Define optimizer module.
optimizer = torch.optim.RMSprop( net.parameters(), lr=args.lr)

def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)

for epoch in range(1, args.epochs+1):
    # Training loop.
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (tac_right, tac_left, _, label) in enumerate(train_loader, 0):

        tac_left = tac_left.to(device)
        tac_right = tac_right.to(device)
        label = label.to(device)
        # Forward pass of the network.
        #print(in_viz.shape)
        out = net.forward(tac_left, tac_right)
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
    train_acc = correct/len(train_loader.dataset)
    
    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    train_acc = 0
    with torch.no_grad():
        for i, (tac_right, tac_left, _, label) in enumerate(train_loader, 0):
            tac_left = tac_left.to(device)
            tac_right = tac_right.to(device)
            label = label.to(device)
            # Forward pass of the network.
            out = net.forward(tac_left, tac_right)
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
        for i, (tac_left, tac_right, _, label) in enumerate(test_loader, 0):
            tac_left = tac_left.to(device)
            tac_right = tac_right.to(device)
            # Forward pass of the network.
            out = net.forward(tac_left, tac_right)
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