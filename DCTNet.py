from resoning import Resoning
from Transformer import EncoderLayers , DecoderLayers
from dcision import ActorNet_policy , CriticNet_values
from memorys import LstmModels
from BackboneEncoder import BackBoneEncoder
import torch
from torchvision import transforms 
from torch import nn
import PIL.Image as img
import numpy as np

class DCTNet: 
    
    '''DCTnet khusus gambar'''
            
    def __init__(self):
        super().__init__()
    
    class Models(nn.Module): 
        
        def __init__(self , hidden_size , nhead , drop , n_flow , n_act):
            super().__init__()
            self.Resoning = Resoning(500 , hidden_size , 2)
            self.encoder = EncoderLayers(hidden_size , nhead , drop)
            self.decoder = DecoderLayers(hidden_size , nhead , drop)
            self.ekstrak = BackBoneEncoder(n_flow)
            self.memorys = LstmModels(hidden_size , n_flow , n_flow , 2)
            self.decision1 = ActorNet_policy(n_flow , 10 , n_act)
            self.decision2 = CriticNet_values(n_flow , 10 , 1)
        def forward(self , in1 , in2): 
            in1 = img.fromarray((in1*255).astype(np.uint8))
            in2 = img.fromarray((in2*255).astype(np.uint8))
            in1_ = transforms.Compose([transforms.Resize((120 , 120)),
                                      transforms.ToTensor()])(in1)
            in1 = torch.unsqueeze(in1_ , 0) / 255.0
            in2_ = transforms.Compose([transforms.Resize((120 ,120)), 
                                       transforms.RandomRotation(25),
                                       transforms.ToTensor()])(in2)
            in2 = torch.unsqueeze(in2_ , 0) / 255.0
            x1 = self.ekstrak(in1)
            x2 = self.ekstrak(in2)
            x1 = self.Resoning(x1)
            print(x1.shape)
            x2 = self.Resoning(x2)
            print(x2.shape)
            x1 = self.encoder(x1)
            x2 = self.decoder(x2)
            x = torch.add(x1 , x2)
            x = self.memorys(x)
            polcy = self.decision1(x)
            values = self.decision2(x)
            return polcy , values           

            