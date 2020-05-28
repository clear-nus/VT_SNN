import torch
import slayerSNN as snn

class EncoderVis(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(EncoderVis, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1   = self.slayer.dense((50, 63, 2), 1024)
        self.fc2   = self.slayer.dense(1024, output_size)
    def forward(self, downsampled):
        spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(downsampled))) # 32, 32, 16
        spikeLayer5 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1))) #  10
        self.spike_trains = [spikeLayer1]
        return spikeLayer5

class EncoderTact(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(EncoderTact, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.fc1   = slayer.dense(156, output_size)
    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp((spikeInput))))
        self.spike_trains = []
        return spikeLayer1

class SlayerMM(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(SlayerMM, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.tactile = EncoderTact(netParams, 50)
        self.vision = EncoderVis(netParams, 10)
        self.slayer = slayer
        self.fc1   = slayer.dense(60, output_size)

    def forward(self, spikeInputTact, spikeInputVis):
        spikeLayer1 = self.tactile(spikeInputTact)
        spikeLayer2 = self.vision(spikeInputVis)
        spikeAll = torch.cat([spikeLayer1, spikeLayer2], dim=1)
        self.spike_trains = [spikeAll] + self.tactile.spike_trains + self.vision.spike_trains

        out = self.slayer.spike(self.slayer.psp(self.fc1(spikeAll)))
        return out