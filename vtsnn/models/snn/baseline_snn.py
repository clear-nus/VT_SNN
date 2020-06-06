import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer

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
    
# class SlayerVisionMLP(torch.nn.Module):
#     def __init__(self, netParams, hidden_size, output_size):
#         super(EncoderVis, self).__init__()
#         self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
#         self.fc1   = self.slayer.dense((50, 63, 2), hidden_size)
#         self.fc2   = self.slayer.dense(hidden_size, output_size)
#     def forward(self, downsampled):
#         spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(downsampled))) # 32, 32, 16
#         spikeLayer5 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1))) #  10
#         self.spike_trains = [spikeLayer1]
#         return spikeLayer5

# class SlayerVisMLP(torch.nn.Module):
#     def __init__(self, netParams, hidden_size, output_size):
#         super(SlayerVisMLP, self).__init__()
#         self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
#         self.fc1 = self.slayer.dense((50, 63, 2), hidden_size)
#         self.fc2 = self.slayer.dense(hidden_size, output_size)

#     def forward(self, downsampled):
#         spike_1 = self.slayer.spike(self.slayer.psp(downsampled)) # 32, 32, 16
#         spike_2 = self.slayer.spike(self.fc1(self.slayer.psp(spike_1))) #  10
#         spike_3 = self.slayer.spike(self.fc2(self.slayer.psp(spike_2)))
#         self.spike_trains = [spike_1, spike_2]
#         return spike_3

class SlayerLoihiMLP(torch.nn.Module):
    '''
    2 layer MLP based on SLAYER used for tactile data
    '''
    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerLoihiMLP, self).__init__()
        self.output_size = output_size
        self.slayer = spikeLayer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spikeLoihi(self.slayer.psp(self.fc1(spike_input)))
        spike_1 = self.slayer.delayShift(spike_1, 1)
        spike_output = self.slayer.spikeLoihi(self.slayer.psp(self.fc2(spike_1)))
        spike_output = self.slayer.delayShift(spike_output, 1)
        self.spike_trains = [spike_1]
        return spike_output
