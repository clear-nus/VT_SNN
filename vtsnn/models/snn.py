import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer


class SlayerMLP(torch.nn.Module):
    """2-layer MLP built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerMLP, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))
        return spike_output


class EncoderVis(torch.nn.Module):
    def __init__(self, params, input_size, output_size):
        super(EncoderVis, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, output_size)

    def forward(self, spike_input):
        spike_output = self.slayer.spike(self.fc1(self.slayer.psp(spike_input)))
        return spike_output


class EncoderTact(torch.nn.Module):
    def __init__(self, params, input_size, output_size):
        super(EncoderTact, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, output_size)

    def forward(self, spike_input):
        spike_output = self.slayer.spike(
            self.fc1(self.slayer.psp((spike_input)))
        )
        return spike_output


class SlayerMM(torch.nn.Module):
    def __init__(
        self,
        params,
        tact_input_size,
        vis_input_size,
        tact_output_size,
        vis_output_size,
        output_size,
    ):
        super(SlayerMM, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.tactile = EncoderTact(params, tact_input_size, tact_output_size)
        self.vision = EncoderVis(params, vis_input_size, vis_output_size)
        self.fc1 = self.slayer.dense(60, output_size)

    def forward(self, spikeInputTact, spikeInputVis):
        spikeLayer1 = self.tactile(spikeInputTact)
        spikeLayer2 = self.vision(spikeInputVis)
        spikeAll = torch.cat([spikeLayer1, spikeLayer2], dim=1)
        
        out = self.slayer.spike(self.slayer.psp(self.fc1(spikeAll)))
        return out
