import numpy as np 
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser
from collections import OrderedDict

# concat_axis = 1 if K.image_data_format() == 'channels_first' else -1  # channel can only be 1 

class SST(torch.nn.Module):

    def __init__(self, input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, \
        include_top=True, attention=True, spatial_attention=True, temporal_attention=True, nb_class=3): 
        super(SST,self).__init__()

        self.specNet = DenseNet(depth=depth_spec, nb_dense_block=nb_dense_block,
                             growth_rate=gr_spec, nb_classes=nb_class, reduction=0.5, bottleneck=True, include_top=False, attention=attention, \
                             spatial_attention=spatial_attention, temporal_attention=temporal_attention)

        self.tempNet = DenseNet(depth=depth_tem, nb_dense_block=nb_dense_block,
                             growth_rate=gr_tem, nb_classes=nb_class, bottleneck=True, include_top=False, subsample_initial_block=True, \
                             attention=attention)

        layers = []
        spec_out = 48
        temp_out = 264 # can be changed accordingly

        layers.append(nn.Linear(spec_out+temp_out, 50))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(50, nb_class))

        # if nb_class == 2:
        #     layers.append(nn.Sigmoid())
        # else:
        #     layers.append(nn.Softmax())

        self.layers = nn.ModuleList(layers)

    def forward(self, spect_input, temp_input):
        spect_output = self.specNet(spect_input)
        temp_output = self.tempNet(temp_input)
        output = torch.cat([spect_output, temp_output], dim=1)
        for layer in self.layers:
            output = layer(output)

        return output

class DenseNet(torch.nn.Module):
    def __init__(self, nb_classes, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax', attention=True, spatial_attention=True, temporal_attention=True): 
        super(DenseNet,self).__init__()
        self.attention = attention

        if reduction != 0.0:
            assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block)'
            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (
                    depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (5, 5, 3)
            initial_strides = (2, 2, 1)
        else:
            initial_kernel = (3, 3, 1)
            initial_strides = (1, 1, 1)

        layers = []
        if subsample_initial_block:
            conv_layer = nn.Conv3d(1, nb_filter, initial_kernel, stride=initial_strides, padding=(2,2,1), bias=False)
        else:
            conv_layer = nn.Conv3d(1, nb_filter, initial_kernel, stride=initial_strides, padding=(1,1,0), bias=False)
        layers.append(("conv1", conv_layer))

        if subsample_initial_block:
            layers.append(("batch1", nn.BatchNorm3d(nb_filter, eps=1.1e-5)))
            layers.append(("active1", nn.ReLU()))
            layers.append(("maxpool", nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), padding=(0,0,1))))
        self.conv_layer = nn.Sequential(OrderedDict(layers))

        if subsample_initial_block:
            initial_width = 8
            initial_hight = 13
        else:
            initial_width = 32
            initial_hight = 5

        layers = []
        for block_idx in range(nb_dense_block - 1):

            if attention:
                layers.append(Attention(nb_filter, initial_width, initial_hight, spatial_attention=spatial_attention, temporal_attention=temporal_attention))

            layers.append(DenseBlock(nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay))  
            nb_filter = nb_filter + growth_rate * nb_layers[block_idx]

            layers.append(Transition(nb_filter, nb_filter, compression=compression, weight_decay=weight_decay))
            nb_filter = int(nb_filter * compression) # 24

            initial_width = int(initial_width / 2) # 16
            initial_hight = int(initial_hight / 2) # 2

            # layers.append(DenseBlock(nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
            #                          dropout_rate=dropout_rate, weight_decay=weight_decay))
            # nb_filter = nb_filter + growth_rate * nb_layers[block_idx]
            # layers.append(Transition(nb_filter, nb_filter, compression=compression, weight_decay=weight_decay))
            # nb_filter = int(nb_filter * compression) # 24

            # initial_width = int(initial_width / 2) # 16
            # initial_hight = int(initial_hight / 2) # 2

            # if attention:
            #     layers.append(Attention(nb_filter, initial_width, initial_hight, spatial_attention=spatial_attention, temporal_attention=temporal_attention))


        layers.append(Attention(nb_filter, initial_width, initial_hight, spatial_attention=spatial_attention, temporal_attention=temporal_attention))
        layers.append(DenseBlock(nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay))  
        nb_filter = nb_filter + growth_rate * nb_layers[block_idx]

        # bt, 24(c), 8, 8, 1
        self.layers = nn.ModuleList(layers)

        final_layers = []

        # final_layers.append(DenseBlock(final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
        #                              dropout_rate=dropout_rate, weight_decay=weight_decay))

        # nb_filter = nb_filter + growth_rate * final_nb_layer

        final_layers.append(nn.BatchNorm3d(nb_filter, eps=1.1e-5))
        final_layers.append(nn.ReLU())
        final_layers.append(nn.AvgPool3d((initial_width, initial_width, initial_hight)))
        self.final_layers = nn.ModuleList(final_layers)

        self.include_top = include_top
        if include_top:
            top_layers = []
            top_layers.append(nn.Linear(nb_filter, nb_classes))

            # if activation == "softmax":
            #     top_layers.append(nn.Softmax())
            # else:
            #     top_layers.append(nn.Sigmoid())
            
            self.top_layers = nn.ModuleList(top_layers)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x =  self.conv_layer(x)

        for layer in self.layers:
            x = layer(x)

        for layer in self.final_layers:
            x = layer(x)
        x = x.view(x.size()[0], -1)
        
        if self.include_top:
            for layer in self.top_layers:
                x = layer(x)
        return x

class DenseBlock(torch.nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False): 
        super(DenseBlock,self).__init__()

        layers = []

        for i in range(nb_layers):
            convLayer = Conv(nb_filter, growth_rate, bottleneck, dropout_rate, weight_decay)
            nb_filter = nb_filter + growth_rate
            layers.append(convLayer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            cb = layer(x)
            x = torch.cat([x, cb], dim=1)
        return x

class Conv(torch.nn.Module):
    def __init__(self, input_channel, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4, add_11_conv=True): 
        super(Conv,self).__init__()

        layers = []

        layers.append(nn.BatchNorm3d(input_channel, eps=1.1e-5))
        layers.append(nn.ReLU())

        if bottleneck:
            inter_channel = nb_filter * 4

            layers.append(nn.Conv3d(input_channel, inter_channel, (1, 1, 1), padding=0, bias=False))
            layers.append(nn.BatchNorm3d(inter_channel, eps=1.1e-5))
            layers.append(nn.ReLU())

        layers.append(nn.Conv3d(inter_channel, nb_filter, (3, 3, 1), padding=(1,1,0), bias=False))

        if add_11_conv:
            layers.append(nn.Conv3d(nb_filter, nb_filter, (1, 1, 1), padding=(0,0,0), bias=False))

        layers.append(nn.Conv3d(nb_filter, nb_filter, (1, 1, 3), padding=(0,0,1), bias=False))
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(torch.nn.Module):
    def __init__(self, input_channel, nb_filter, compression=1.0, weight_decay=1e-4): 
        super(Transition,self).__init__()

        layers = []

        layers.append(nn.BatchNorm3d(input_channel, eps=1.1e-5))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(input_channel, int(nb_filter * compression), (1, 1, 1), padding=0, bias=False))
        layers.append(nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Attention(torch.nn.Module):
    def __init__(self, input_channel, initial_width, initial_hight, spatial_attention=True, temporal_attention=True): 
        super(Attention,self).__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention

        nbSpatial = int(initial_width) * int(initial_width)
        self.spatial_pool = nn.AvgPool3d(kernel_size=[1, 1, initial_hight])
        self.spatail_dense = nn.Linear(nbSpatial, nbSpatial)

        nbTemporal = initial_hight
        self.temp_pool = nn.AvgPool3d(kernel_size=[initial_width, initial_width, 1])
        self.temp_dense = nn.Linear(initial_hight, initial_hight)

    def forward(self, _input):
        x = torch.mean(_input, dim=1)
        x = x.unsqueeze(1)

        nbSpatial = x.size()[2] * x.size()[3]
        nbTemporal = x.size()[-1]

        if self.spatial_attention:
            spatial = self.spatial_pool(x)
            spatial = spatial.view(-1, nbSpatial)
            spatial = self.spatail_dense(spatial)
            spatial = F.sigmoid(spatial) # 
            spatial = spatial.view(x.size()[0], 1, x.size()[2], x.size()[3], 1)

            tem = _input * spatial # may check here, questions

        if self.temporal_attention:
            temporal = self.temp_pool(x)
            temporal = temporal.view(-1, nbTemporal)
            temporal = self.temp_dense(temporal)
            temporal = F.sigmoid(temporal) #
            temporal = temporal.view(x.size()[0], 1, 1, 1, x.size()[-1])

            tem = temporal * tem

        return tem