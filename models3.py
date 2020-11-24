






# new Spatial attention model skip connection

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 16, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(16, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)




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
        self.sa1 = SpatialAttention()
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
        self.sa2 = SpatialAttention()
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
        self.sa3 = SpatialAttention()
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        out = x
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = x + out
        x = (self.sa1(x) * x) + x
        #Conv2
        x = self.layer5(x)
        out = x
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        x = x + out
        x = (self.sa2(x) * x) + x
        #Conv3
        x = self.layer9(x) 
        out = x   
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        x = x + out
        x = (self.sa3(x) * x) + x
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
        
        self.sa4 = SpatialAttention()
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
        
        self.sa5 = SpatialAttention()
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
        
        self.sa6 = SpatialAttention()
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        

    def forward(self,x):        
        #Deconv3
        out = x
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = x + out
        x = (self.sa4(x) * x) + x
        x = self.layer16(x)                
        #Deconv2
        out = x
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        #x = self.ca5(x) * x
        x = (self.sa5(x) * x) + x
        x = x + out
        x = self.layer20(x)
        #Deconv1
        out = x
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = x + out
        x = (self.sa6(x) * x) + x
        x = self.layer24(x)
        return x


