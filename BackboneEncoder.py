import torch 
from torch import nn 

class VGGBlock(nn.Module): 
    def __init__(self , c_in , c_out , Kernels ,  Stride , leaky , drop , poolK):
        super(VGGBlock , self).__init__()
        self.conv = nn.Conv2d(in_channels= c_in , out_channels=c_out,
                              kernel_size=Kernels , stride=Stride)
        self.bctn = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(drop)
        self.leak = nn.LeakyReLU(leaky)
        self.Mpool = nn.MaxPool2d(poolK)
    def forward(self , inputs): 
        x1 = self.conv(inputs)
        x = self.leak(x1)
        x2 = self.bctn(x)
        com = torch.add(x1 , x2)
        x = self.Mpool(com)
        x = self.drop(x)
        return x 

class BackBoneEncoder(nn.Module): 
    def __init__(self , nFlow):
        super(BackBoneEncoder , self).__init__()
        self.VGGB1 = VGGBlock(3 , 32 , 3 , 1 , 0.5 , 0.5 , 1)
        self.VGGB2 = VGGBlock(32 , 32 , 3 , 1 , 0.5 , 0.5 , 1)
        self.VGGB3 = VGGBlock(32 , 32 , 3 , 1 , 0.5 , 0.5 , 2)
        self.VGGB4 = VGGBlock(32 , 64 , 3 , 1 , 0.5 , 0.5 , 1)
        self.VGGB5 = VGGBlock(64 , 128 , 3 , 1 , 0.5 , 0.5 , 3)
        self.VGGB6 = VGGBlock(128 , 32 , 3 , 1 , 0.5 , 0.5 , 3)
        self.flatern = nn.Flatten()
        self.lin1 = nn.Linear(4608 , 120)
        self.lin2 = nn.Linear(120 , nFlow)
    def forward(self, inputs): 
        x = self.VGGB1(inputs)
        x = self.VGGB4(x)
        x = self.VGGB5(x)
        x = self.VGGB6(x)
        x = self.flatern(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x 
