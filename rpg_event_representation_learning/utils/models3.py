import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = '/home/tasbolat/some_python_examples/VT_SNN/rpg_event_representation_learning/utils/quantization_layer_init/trilinear_init.pth'
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
#         path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
#         if isfile(path):
#             state_dict = torch.load(path)
#             self.load_state_dict(state_dict)
#         else:
#             self.init_kernel(num_channels)


    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        #print('vox shape:', vox.shape)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        
        #print(idx_before_bins)

        for i_bin in range(C-1): # C
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            #print(t-i_bin, values.shape)

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            #print('within batch:', i_bin, idx, values)
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox

class SimpleNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=30, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(30, 80, 3)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(1200, num_classes)
    def forward(self, x):
        conv_lay1 = self.conv1(x)
        #print(conv_lay1.shape)
        conv_lay1 = self.relu1(conv_lay1)
        conv_lay2 = self.conv2(conv_lay1)
        conv_lay2 = self.relu2(conv_lay2)
        #print(conv_lay2.shape)
        conv_lay2 = torch.flatten(conv_lay2, 1)
        #print(conv_lay2.shape)
        out_lay = self.fc(conv_lay2)
        return out_lay
    
class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer1 = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.quantization_layer2 = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        

        # replace fc layer and first convolutional layer
        input_channels = 2*voxel_dimension[0]
        # simple classifier consisting of conv layers for tactile data
        self.classifier = SimpleNet(input_channels, num_classes)

    def forward(self, x1):
        vox1 = self.quantization_layer1.forward(x1)
        out = self.classifier.forward(vox1)
        return out, vox1


