"""SNN models compatible with the Loihi."""

from pathlib import Path
import numpy as np
import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer


class SlayerLoihiMLP(torch.nn.Module):
    """
    2 layer MLP based on SLAYER, using the Loihi neuron model.
    """

    def __init__(
        self, params, input_size, hidden_size, output_size, quantize=True
    ):
        super(SlayerLoihiMLP, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.quantize = quantize
        self.slayer = spikeLayer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(
            input_size, hidden_size, quantize=self.quantize
        )
        self.fc2 = self.slayer.dense(
            hidden_size, output_size, quantize=self.quantize
        )

    def forward(self, spike_input):
        spike_1 = self.slayer.spikeLoihi(self.fc1(spike_input))
        spike_1 = self.slayer.delayShift(spike_1, 1)
        spike_output = self.slayer.spikeLoihi(self.fc2(spike_1))
        spike_output = self.slayer.delayShift(spike_output, 1)
        self.spike_trains = [spike_1]
        return spike_output

    def quantize(self, epoch, checkpoint_dir):
        fc1_weights = (
            snn.utils.quantize(self.fc1.weight, 2)
            .flatten()
            .cpu()
            .data.numpy()
            .reshape(self.hidden_size, -1)
        )
        fc2_weights = (
            snn.utils.quantize(self.fc2.weight, 2)
            .flatten()
            .cpu()
            .data.numpy()
            .reshape(self.output_size, -1)
        )

        np.save(
            Path(checkpoint_dir) / f"{epoch:03d}-fc1.npy", fc1_weights,
        )
        np.save(
            Path(checkpoint_dir) / f"{epoch:03d}-fc2.npy", fc2_weights,
        )


class SlayerLoihiMM(torch.nn.Module):
    def __init__(
        self,
        params,
        tact_input_size,
        vis_input_size,
        tact_output_size,
        vis_output_size,
        output_size,
    ):
        super(SlayerLoihiMM, self).__init__()
        self.tact_output_size = tact_output_size
        self.vis_output_size = vis_output_size
        self.output_size = output_size
        self.slayer = spikeLayer(params["neuron"], params["simulation"])

        self.tact_fc = self.slayer.dense(tact_input_size, tact_output_size)
        self.vis_fc = self.slayer.dense(vis_input_size, vis_output_size)
        self.combi = self.slayer.dense(
            tact_output_size + vis_output_size, output_size
        )

    def forward(self, spikeInputTact, spikeInputVis):
        tact_int = self.slayer.spikeLoihi(self.tact_fc(spikeInputTact))
        tact_int = self.slayer.delayShift(tact_int, 1)
        vis_int = self.slayer.spikeLoihi(self.vis_fc(spikeInputVis))
        vis_int = self.slayer.delayShift(vis_int, 1)
        combi = torch.cat([tact_int, vis_int], dim=1)
        out = self.slayer.spikeLoihi(self.combi(combi))
        out = self.slayer.delayShift(out, 1)

        return out

    def quantize(self, epoch, checkpoint_dir):
        tact_fc_weights = (
            snn.utils.quantize(self.tact_fc.weight, 2)
            .flatten()
            .cpu()
            .data.numpy()
            .reshape(self.tact_output_size, -1)
        )
        vis_fc_weights = (
            snn.utils.quantize(self.vis_fc.weight, 2)
            .flatten()
            .cpu()
            .data.numpy()
            .reshape(self.vis_output_size, -1)
        )
        combi_weights = (
            snn.utils.quantize(self.combi.weight, 2)
            .flatten()
            .cpu()
            .data.numpy()
            .reshape(self.output_size, -1)
        )

        np.save(
            Path(checkpoint_dir) / f"{epoch:03d}-tact.npy", tact_fc_weights,
        )
        np.save(
            Path(checkpoint_dir) / f"{epoch:03d}-vis.npy", vis_fc_weights,
        )
        np.save(
            Path(checkpoint_dir) / f"{epoch:03d}-combi.npy", combi_weights,
        )
