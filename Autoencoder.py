import torch
import torch.nn as nn

class Convolutional_Block(nn.Module):
    def __init__(self,in_channel,out_channel,factor_scale,kernel_size,encode=True,isAdjust=False):
        if (kernel_size - factor_scale) % 2 != 0 and encode:
            raise ValueError("kernel_size - factor_scale must be even")
        super(Convolutional_Block,self).__init__()
        if isAdjust:
            self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size,1,(kernel_size - 1)/2)
        else:
            if encode:
                self.conv2 = nn.Conv2d(in_channel, out_channel,kernel_size,factor_scale,(kernel_size - factor_scale)/2)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1,1)
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_channel)

    def forward(x,self):
        if self.encode:  
            H = self.bnorm(self.relu(self.conv1(x)))
            return self.conv2(H)
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.channel_decompostion_up = (16,32,64,128,256)
        self.reformulating_layer_factor = 4
        

        

