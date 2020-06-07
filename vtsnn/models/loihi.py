import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer


class SlayerLoihiMLP(torch.nn.Module):
    """2-layer MLP based on SLAYER used for tactile data. To be trained and run on the Loihi."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerLoihiMLP, self).__init__()
        self.output_size = output_size
        self.slayer = spikeLayer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)
        self.spike_trains = None

    def forward(self, spike_input):
        spike_1 = self.slayer.spikeLoihi(self.slayer.psp(self.fc1(spike_input)))
        spike_1 = self.slayer.delayShift(spike_1, 1)
        spike_output = self.slayer.spikeLoihi(
            self.slayer.psp(self.fc2(spike_1))
        )
        spike_output = self.slayer.delayShift(spike_output, 1)
        self.spike_trains = [spike_1]
        return spike_output
