import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch import nn
import matplotlib.pyplot as plt

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

device = torch.device("cuda")
writer = SummaryWriter(".")

train_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"train_80_20_{args.sample_file}.txt", output_size=args.output_size
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt", output_size=args.output_size
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)

class MLP_LSTM(nn.Module):
    def __init__(self):
        super(MLP_LSTM, self).__init__()
        self.input_size = 156
        self.hidden_dim = args.hidden_size

        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)
        self.fc = nn.Linear(self.hidden_dim, args.output_size)

    def forward(self, input_data):
        gru_in = input_data
        gru_in = gru_in.permute(1,0,2)
        gru_out, self.hidden = self.gru(gru_in)
        y_pred = self.fc(gru_out[-1, :, :])
        return y_pred

def _save_model(epoch, loss):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"weights-{epoch:03d}-{loss:0.3f}.pt"
    )
    torch.save(net.state_dict(), checkpoint_path)


net = MLP_LSTM().to(device)
# Create snn loss instance.
criterion = nn.CrossEntropyLoss()
# Define optimizer module.
optimizer = torch.optim.RMSprop(
    net.parameters(), lr=args.lr) # 0.0001

for epoch in range(1, args.epochs+1):
    # Training loop.
    net.train()
    correct = 0
    batch_loss = 0
    train_acc = 0
    for i, (in_tact, _, _, label) in enumerate(train_loader, 0):
        in_tact = in_tact.to(device)
        in_tact = in_tact.squeeze()
        in_tact = in_tact.permute(0,2,1)
        label = label.to(device)
        # Forward pass of the network.
        out_tact = net.forward(in_tact)
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
    train_acc = correct/len(train_loader.dataset)
    writer.add_scalar("loss/train", batch_loss/len(train_loader.dataset), epoch)
    writer.add_scalar("acc/train", train_acc, epoch)

    # testing
    net.eval()
    correct = 0
    batch_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (in_tact, _, _, label) in enumerate(test_loader, 0):
            in_tact = in_tact.to(device)
            in_tact = in_tact.squeeze()
            in_tact = in_tact.permute(0,2,1)

            # Forward pass of the network.
            out_tact = net.forward(in_tact)
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
