import torch
import torch.nn as nn
from functools import partial
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np

from .intialisation import init_weights

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)      
        

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, 
                 *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels, momentum=0.999)) \
            if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
           nn.BatchNorm2d(out_channels, momentum=0.999))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, 
                    bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
    

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, 
                     stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, stage, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        self.stage = stage
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        
        if self.stage == 0:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.blocks_sizes[0], momentum=0.999),
                activation_func(activation),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.blocks = ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                            block=block,*args, **kwargs)
        
        else:
            (in_channels, out_channels) = self.in_out_block_sizes[stage-1]
            n = depths[stage]
            self.blocks = ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 

    def forward(self, x):
        if self.stage == 0:
            x = self.gate(x)
        stage_output = self.blocks(x)
        return stage_output


class ResNet(nn.Module):
    
    def __init__(self, stage, resnet_type, in_channels, *args, **kwargs):
        super().__init__()

        if resnet_type == 'resnet18':
            self.encoder = ResNetEncoder(stage, in_channels, block=ResNetBasicBlock, 
                                         depths=[2, 2, 2, 2], *args, **kwargs)
        elif resnet_type == 'resnet34':
            self.encoder = ResNetEncoder(stage, in_channels, block=ResNetBasicBlock, 
                                         depths=[3, 4, 6, 3], *args, **kwargs)
        elif resnet_type == 'resnet50':
            self.encoder = ResNetEncoder(stage, in_channels, block=ResNetBottleNeckBlock, 
                                         depths=[3, 4, 6, 3], *args, **kwargs)
        elif resnet_type == 'resnet101':
            self.encoder = ResNetEncoder(stage, in_channels, block=ResNetBottleNeckBlock, 
                                         depths=[3, 4, 23, 3], *args, **kwargs)
        elif resnet_type == 'resnet152':
            self.encoder = ResNetEncoder(stage, in_channels, block=ResNetBottleNeckBlock, 
                                         depths=[3, 8, 36, 3], *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return x


class ConvLayer2D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(ConvLayer2D, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k_size,
                      padding=padding, stride=stride, bias=True),
            nn.ReLU(inplace=True)
        )
                                       
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv_unit(inputs)
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self, fv_dim_cnn, fv_dim_hidden):
        super(AutoEncoder, self).__init__()

        self.ae = nn.Sequential(
            nn.Linear(fv_dim_cnn, fv_dim_hidden, bias=True),
            nn.Linear(fv_dim_hidden, fv_dim_cnn, bias=False)
        )
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        fv = self.ae[0](inputs)
        recon = self.ae(inputs)

        return fv, recon


class OutputMLP(nn.Module):
    def __init__(self, fv_dim_hidden, n_out_features):
        super(OutputMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(fv_dim_hidden, fv_dim_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fv_dim_hidden, fv_dim_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fv_dim_hidden, n_out_features, bias=False)
            )

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):        
        return self.mlp(inputs)


