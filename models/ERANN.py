import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        assert in_channels // reduction != 0
        self.SE = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // reduction, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, (1, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.SE(x)
    
    
class ARB(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 stride, droprate, se_included):
        super(ARB, self).__init__()
        z_1_2 = {1, 2, (1, 1), (1, 2), (2, 1), (2, 2)} 
        z_4 = {(4, 1), (4, 2)}
        z_4_1 = {(1, 4), (2, 4)}
        if stride in z_1_2:
            kernel_size1, kernel_size2= (3, 3), (3, 3)
            padd1, padd2 = (1, 1), (1, 1)
        elif stride in z_4:
            kernel_size1, kernel_size2 = (6, 3), (5, 3)
            padd1, padd2 = (2, 1), (2, 1)
        elif stride in z_4_1: 
            kernel_size1, kernel_size2 = (3, 6), (3, 5)
            padd1, padd2 = (1, 2), (1, 2)

        self.droprate = droprate
        if self.droprate != 0:
            self.dropout = nn.Dropout(self.droprate)
        
        self.se_included = se_included
        if se_included:
            self.se_block = SEBlock(out_channels, reduction=8)
            
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.001),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
        self.layer2 = nn.Conv2d(
            in_channels, out_channels, kernel_size1, stride, 
            padding=padd1, bias=True
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=0.001),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
        self.layer4 = nn.Conv2d(
            out_channels, out_channels, kernel_size2, (1, 1),
            padding=padd2, bias=True
        )
        self.layerRes = nn.Conv2d(in_channels, out_channels, (1, 1), 
                                  stride, padding=(0, 0), bias=True)
                                  
    def forward(self, x):
        xRes = self.layerRes(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.droprate != 0:
            x = self.dropout(x)
            
        x = self.layer4(x)
        
        if self.se_included:
            x = self.se_block(x)
            
        return x + xRes
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, 
                 droprate, se_included=False, activation_last=True,
                 stride_only_first=True):
        super(BasicBlock, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True) 
        self.activation_last = activation_last
        self.stride_only_first = stride_only_first
        self.layer = self.make_layer(in_channels, out_channels, stride, droprate, se_included)

    def make_layer(self, in_channels, out_channels, stride, droprate, se_included):
        layer = []
        
        for i in range(4):
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            # Изначально
            if self.stride_only_first:
                strd = (1, 1) if i != 0 else stride
            else:
                strd = stride

            in_chnnls = out_channels if i != 0 else in_channels
            layer.append(ARB(in_chnnls, out_channels, strd, droprate, se_included))

            if i != 3 or self.activation_last:
                layer.append(self.activation)

        return nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)
    

class ClassifyBlock(nn.Module):
    def __init__(self, num_classes, num_channels, activation, droprate=0.):
        super(ClassifyBlock, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((1 ,1))

        assert activation in {'sigmoid', 'softmax', None}
        # 'sigmoid' or activation == 'softmax' or activation == None
        
        FC = [
              nn.Dropout(droprate),
              nn.Linear(in_features=num_channels, out_features=num_channels, bias=True),
              nn.LeakyReLU(negative_slope=0.01, inplace=True),
              nn.Dropout(droprate),
              nn.Linear(in_features=num_channels, out_features=num_classes, bias=True),
        ]
        
        if activation == 'softmax':
            FC.append(nn.Softmax(dim=-1))
        elif activation == 'sigmoid':
            FC.append(nn.Sigmoid())

        self.FC = nn.Sequential(*FC)


    def forward(self, x):
        x = self.adaptive_avg_pool(x) + self.adaptive_max_pool(x)
        x = self.flatten(x)
        x = self.FC(x)
        
        return x
      

class ERANN(nn.Module): # each block has dropout for making overfit promlem less
    def __init__(self, num_classes, num_blocks=5, widen=10, Xstrides=[2, 2, 2, 2], 
                 droprate=0.0, se_included=False, activation=None, started_channels=1, stride_only_first=True):
        super(ERANN, self).__init__()
        self.droprate = droprate
        # Usually, [1, 8, 16, 32, 64, 128] and each except first is multiplied on widen 
        nChannels = [started_channels] + [2**(i + 3) * widen for i in range(num_blocks + 1)]
        # if num_blocks > 1:
        #     Xstrides = Xstrides + [2]
    
        nStrides = [(2, Xstrides[i]) for i in range(0, len(Xstrides))]
        #nStrides = [(1, 1)] + [(2, Xstrides[i]) for i in range(0, len(Xstrides))] 

        assert 1 <= num_blocks <= 5
        assert len(Xstrides) == num_blocks

        self.BasicStage = []

        for i in range(num_blocks):
            if i + 1 == num_blocks:
                activation_last = False
            else:
                activation_last = True

            self.BasicStage.append(
                BasicBlock(
                    in_channels=nChannels[i], out_channels=nChannels[i + 1], 
                    stride=nStrides[i], droprate=droprate, se_included=se_included,
                    activation_last=activation_last, stride_only_first=stride_only_first
            ))

        self.BasicStage = nn.Sequential(*self.BasicStage)
        self.ClassifyStage = ClassifyBlock(
            num_classes=num_classes, num_channels=nChannels[num_blocks], 
            activation=activation, droprate=droprate
        )

    def forward(self, x):
        x = self.BasicStage(x)
        x = self.ClassifyStage(x)
        
        return x
        