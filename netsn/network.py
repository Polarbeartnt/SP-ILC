from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np
from netsn.netblock import Resblock_body, BasicConv, yolo_head, Upsample, FcLayer

class ForHead(nn.Module):
    def __init__(self, M):
        super(ForHead, self).__init__()
        self.M = M
        self.inlayer = FcLayer(M,M)
        self.translayer = FcLayer(M,1024)
        # Zero init for input layer, which will be freezed before seeing expdata
        self.inlayer.fc.weight.data = torch.eye(M)
        self.inlayer.fc.bias.data = torch.zeros_like(self.inlayer.fc.bias.data)
    
    def forward(self, x): # batch, channel, [size]
        x = self.inlayer(x) # b, 1, M
        x = self.translayer(x) # b, 1, 1024
        shape = np.append([x.shape[0],1], [32,32]) 
        x = x.reshape(tuple(shape)) # b, 1, 32, 32
        return x
    
    def reinit(self):
        self.inlayer = FcLayer(self.M, self.M) 

class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        
        self.resblock_body1 =  Resblock_body(1, 64)
        self.resblock_body2 =  Resblock_body(64, 128)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x, _    = self.resblock_body1(x) # (b, 64, 32, 32), (b, 32, 32, 32)
        feat0 = x
        x = self.maxpool(x) # b, 128, 16, 16
        x, feat1    = self.resblock_body2(x) # (b, 128, 16, 16), (b, 64, 16, 16)
        return x,feat0,feat1

#---------------------------------------------------#
#   Net body
#---------------------------------------------------#
class MyNetBody(nn.Module):
    def __init__(self, num_anchors, num_classes, M):
        super(MyNetBody, self).__init__()

        self.forhead = ForHead(M)
        self.backbone = BackBone()

        self.conv1 = BasicConv(128,64,1)
        self.yolo_head1 = yolo_head([64, num_anchors * (5 + num_classes)], 64)
        self.upsample = Upsample(64,64)
        self.conv2 = BasicConv(128,64,1)
        self.yolo_head2 = yolo_head([64, num_anchors * (5 + num_classes)], 64)

        self.graphnet = nn.Sequential(
            Upsample(64,32),
            BasicConv(32,16,3),
            BasicConv(16,1,1))

    def forward(self, x):
        x = self.forhead(x)
        forfig = x
        # 16,16
        x, feat0, feat1 = self.backbone(x)
        x = self.conv1(x)
        yolo1 = self.yolo_head1(x)

        # 32,32
        feat1 = self.upsample(feat1)
        x = torch.cat([feat0,feat1], axis=1)
        x = self.conv2(x)
        yolo2 = self.yolo_head2(x)

        # 64,64
        graph = self.graphnet(x)
        graph = graph.reshape(tuple([graph.shape[0], 64, 64]))
        
        return yolo1, yolo2, graph, forfig

class TrainANet(nn.Module):
    def __init__(self, num_anchors, num_classes, M):
        super(TrainANet, self).__init__()
        self.M = M

        #self.pattern = HardBinaryConv(1, M, kernel_size=64, padding=0)
        self.pattern = nn.Conv2d(1,M,64)
        self.mainnet = MyNetBody(num_anchors, num_classes, M)

    def forward(self, x): # Warning: inputs should be image array rather than s series
        
        # Reshape arrays to images
        x = torch.reshape(x, (x.shape[0], 1, 64, 64))

        # Simulate Binary Patterns
        x = self.pattern(x)

        # Reshape to S as input
        x = torch.reshape(x, (x.shape[0], self.M))

        # Normalize
        mins = torch.min(x.detach(), dim=1, keepdim=True)[0]
        maxs = torch.max(x.detach(), dim=1, keepdim=True)[0]
        x = (x - mins)/(maxs-mins)
        
        # return outputs and patterns
        _1, _2, _3, _4 = self.mainnet(x)
        return _1, _2, _3, _4, self.pattern.weight

class FTNet(nn.Module):
    def __init__(self, M):
        super(FTNet, self).__init__()
        self.input = FcLayer(M,1024)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32,32,5,padding=2,stride=2),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64,128,5,padding=2,stride=2),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,64,5,padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,1,3,padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
        self.output = nn.Linear(1024,M)

    def forward(self, x):
        x = self.input(x)
        x = x.reshape((x.shape[0],1,32,32))
        x = self.conv1(x)
        feat1 = x
        x = self.down1(x)
        x = self.conv2(x)
        feat2 = x
        x = self.down2(x)
        x = self.conv3(x)
        
        x = self.up1(x)
        x = torch.cat([x,feat2],axis=1)
        x = self.conv4(x)
        x = self.up2(x)
        x = torch.cat([x,feat1],axis=1)
        x = self.conv5(x)
        x = x.reshape((x.shape[0],1024))
        x = self.output(x)
        

        #min = torch.min(x, dim=1, keepdim=True)[0]
        #max = torch.max(x, dim=1, keepdim=True)[0]
        #x = (x - min)/(max-min)
        return x
