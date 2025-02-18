import torch 
from torch import nn 

class ActorNet_policy(nn.Module): 
    
    def __init__(self , inputs_size , hidden_size, out_size):
        super(ActorNet_policy , self).__init__()
        self.inl = nn.Linear(inputs_size , hidden_size)
        self.lin1 = nn.Linear(hidden_size , hidden_size)
        self.lin2 = nn.Linear(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size , out_size)
        self.lnorms = nn.LayerNorm(hidden_size)
        self.soft = nn.LogSoftmax(-1)
        self.drop = nn.Dropout(0.5)
    def forward(self, inputs): 
        x = self.inl(inputs)
        x = self.drop(x)
        x1 = self.lin1(x)
        x = self.lin2(x)
        x = self.drop(x)
        x2 = self.lnorms(x)
        x = torch.add(x1 , x2)
        x = (self.soft(self.out(x)))
        return x 
    
class CriticNet_values(nn.Module): 
    
    def __init__(self , inputs_size , hidden_size , out_size):
        super(CriticNet_values , self).__init__()
        self.linin = nn.Linear(inputs_size, hidden_size)
        self.hid = nn.Linear(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size , out_size)
        self.rel = nn.LeakyReLU(0.05)
        self.norms = nn.LayerNorm(hidden_size)
        self.tan = nn.Tanh()
    def forward(self , inputs): 
        x = self.linin(inputs)
        x = self.rel(x)
        x = self.hid(x)
        x = self.norms(x)
        x = self.tan(self.out(x))      
        return x   
