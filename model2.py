

# Spatial attention model skip connection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import math
import spectral
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class ChannelGate(nn.Module):
    #def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types='var',gamma = 2, b = 1):   #RAM
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     #nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.ReLU(),
        #     #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #     )
        t = int(abs((math.log(gate_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k, padding = int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid() 
        self.pool_types = pool_types
        ######
        #self.projection = nn.Conv2d(gate_channels, gate_channels, kernel_size=1, stride=1)
        ######
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y
        # if self.pool_types=='var':
        #     var_pool = torch.var(x.view(x.shape[0], x.shape[1], 1, -1,), dim=3, keepdim=True, unbiased=False)
        #     #print('var_pool', var_pool.size())
        #     channel_att_sum = self.mlp( var_pool )
        #     #print('att_sum', channel_att_sum.size())
        
        # #statistics.histogram_ch(channel_att_sum)

        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # #scale = channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        # ######
        # #proj = self.projection(x)
        # ######
        # return scale           #RAM
        # #return x*scale          #CBAM
        # #return x*scale, scale
        # #return proj * scale



class SpatialGate(nn.Module):
    #def __init__(self):
    def __init__(self, gate_channels):
        super(SpatialGate, self).__init__()
        #self.spatial = nn.Sequential(spectral.SpectralNorm(nn.Conv2d(gate_channels, gate_channels, kernel_size=5, padding=2)))
        self.spatial = nn.Sequential(utils.spectral_norm(nn.Conv2d(gate_channels, gate_channels, kernel_size=5, padding=2)))
        '''
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicBlock(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        #self.spatial = BasicBlock(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        '''
    def forward(self, x):
        scale = self.spatial(x)
        #scale = F.sigmoid(scale)
        #statistics.histogram(scale)
        return scale
        #return x
        #return x * scale
        '''
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
        '''

class CBAM(nn.Module):
	######
    #def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types='var', no_spatial=False):         #RAM
	######
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        ######
		#if not no_spatial:
        #   self.SpatialGate = SpatialGate()
        if not no_spatial:
            #if SpatialType == 'FirstOrder':
            self.SpatialGate = SpatialGate(gate_channels)
            #if SpatialType == 'SecondOrder':
            #    self.SpatialGate = Self_Attn(gate_channels, 'relu')
		######
    def forward(self, x):
        ######
        #x_out = self.ChannelGate(x)                     #CBAM
        scale_channel = self.ChannelGate(x)            #RAM
        
        #x_out, scale = self.ChannelGate(x)
        ######
        if not self.no_spatial:
            scale_spatial = self.SpatialGate(x)        #RAM
            #x_out = self.SpatialGate(x_out)             #CBAM
        #print('channel', scale_channel.size())
        #print('spatial', scale_spatial.size())
        scale = scale_channel * scale_spatial          #RAM
        #print(scale)
        #statistics.histogram(scale)
        ######
        x_out = F.sigmoid(scale)                       #RAM
        #x_out = x*scale                                #RAM
        ######
        return x_out
        #return x_out, scale
        ######


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #self.ca1 = ChannelAttention(32)
        #self.sa1 = SpatialAttention()
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #self.ca2 = ChannelAttention(64)
        #self.sa2 = SpatialAttention()
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #self.ca3 = ChannelAttention(128)
        #self.sa3 = SpatialAttention()
        self.cbam = CBAM(128)
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        out = x
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = x + out
        #x = (self.sa1(x) * x) + x
        #Conv2
        x = self.layer5(x)
        out = x
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = x + out
        #x = (self.sa2(x) * x) + x
        #Conv3
        x = self.layer9(x) 
        out = x   
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        x = x + out
        #x = (self.sa3(x) * x) + x
        x = (self.cbam(x) * x) + x
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
        #self.sa4 = SpatialAttention()
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        
        #self.sa5 = SpatialAttention()
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        
        #self.sa6 = SpatialAttention()
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.cbam = CBAM(128)
        

    def forward(self,x):        
        #Deconv3
        out = x
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = x + out
        x = (self.cbam(x) * x) + x
        x = self.layer16(x)                
        #Deconv2
        out = x
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        #x = self.ca5(x) * x
        #x = (self.sa5(x) * x) + x
        x = x + out
        x = self.layer20(x)
        #Deconv1
        out = x
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = x + out
        #x = (self.sa6(x) * x) + x
        x = self.layer24(x)
        return x


