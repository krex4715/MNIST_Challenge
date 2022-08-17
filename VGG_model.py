import torch
import torch.nn as nn


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s, ceil_mode=True)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out, keep_prob, xavier=True):
    linear = nn.Linear(size_in, size_out)
    if xavier :
        torch.nn.init.xavier_uniform_(linear.weight)
    layer = nn.Sequential(
        linear,
        nn.BatchNorm1d(size_out),
        nn.ReLU(),
        nn.Dropout(p = 1 - keep_prob)
    )
    return layer



class MyModule(nn.Module):
    def __init__(self, keep_prob = 0.5, xavier = True):
        super(MyModule, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        channels = [32, 64, 128]

        self.layer1 = vgg_conv_block([3, channels[0]], [channels[0],channels[0]], [3,3], [0,0], 2, 2)
        self.layer2 = vgg_conv_block([channels[0],channels[1]], [channels[1],channels[1]], [3,3], [0,0], 2, 2)
        self.layer3 = vgg_conv_block([channels[1],channels[2]], [channels[2],channels[2]], [3,3], [1,0], 2, 2)

        # FC layers
        self.layer4 = vgg_fc_layer(channels[2], 1024, keep_prob = keep_prob,xavier = xavier)
        self.layer5 = vgg_fc_layer(1024, 800, keep_prob = keep_prob, xavier= xavier)

        # Final layer
        self.layer6 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        out = self.layer1(x)
        out = self.layer2(out)
        vgg16_features = self.layer3(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
