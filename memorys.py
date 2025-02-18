import torch 
from torch import nn 

class LstmModels(nn.Module):
    def __init__(self , inputs_size , hidden , out_size , n_layes = 1, batch_fitst = True):
        super(LstmModels, self).__init__()
        self.Lstm = nn.LSTM(inputs_size , hidden , n_layes , batch_first=batch_fitst)
        self.hidd = hidden
        self.nlayer = n_layes
        self.lin = nn.Linear(hidden , out_size)
        
    def forward(self , inputs:torch.Tensor): 
        h0 = torch.zeros((self.nlayer , inputs.shape[0] , self.hidd))
        c0 = torch.zeros((self.nlayer , inputs.shape[0] , self.hidd))
        x , _ = self.Lstm(inputs , (h0 , c0))
        x = self.lin(x[: , -1 , :])
        return x 