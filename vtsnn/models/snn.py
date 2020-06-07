import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer


class SlayerMLP(torch.nn.Module):
    """
    2-layer MLP based on SLAYER used for tactile data."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerMLP, self).__init__()
        self.output_size = output_size
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)
        self.spike_trains = None

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))
        self.spike_trains = [spike_1]
        return spike_output


class EncoderVis(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(EncoderVis, self).__init__()
        self.slayer = snn.layer(netParams["neuron"], netParams["simulation"])
        self.fc1 = self.slayer.dense((50, 63, 2), output_size)
        self.spike_trains = None

    def forward(self, downsampled):
        spike_output = self.slayer.spike(self.fc1(self.slayer.psp(downsampled)))
        self.spike_trains = []
        return spike_output


class EncoderTact(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(EncoderTact, self).__init__()
        slayer = snn.layer(netParams["neuron"], netParams["simulation"])
        self.slayer = slayer
        self.fc1 = slayer.dense(156, output_size)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp((spikeInput))))
        self.spike_trains = []
        return spikeLayer1


class SlayerMM(torch.nn.Module):
    def __init__(self, netParams, output_size):
        super(SlayerMM, self).__init__()
        slayer = snn.layer(netParams["neuron"], netParams["simulation"])
        self.tactile = EncoderTact(netParams, 50)
        self.vision = EncoderVis(netParams, 10)
        self.slayer = slayer
        self.fc1 = slayer.dense(60, output_size)

    def forward(self, spikeInputTact, spikeInputVis):
        spikeLayer1 = self.tactile(spikeInputTact)
        spikeLayer2 = self.vision(spikeInputVis)
        spikeAll = torch.cat([spikeLayer1, spikeLayer2], dim=1)
        self.spike_trains = (
            [spikeAll] + self.tactile.spike_trains + self.vision.spike_trains
        )

        out = self.slayer.spike(self.slayer.psp(self.fc1(spikeAll)))
        return out
