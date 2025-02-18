import torch 
from torch import nn 

class Resoning(nn.Module): 
    def __init__(self , emeben1 , emeben2 , round:int):
        super(Resoning , self).__init__()
        self.A = nn.Embedding(emeben1 , emeben2)
        self.C = nn.Embedding(emeben1 , emeben2)
        self.U = nn.Embedding(emeben1 , emeben2)
        self.Lnorms = nn.LayerNorm(emeben2)
        self.out = nn.Linear(emeben2 , emeben2)
        self.act = nn.Softmax(dim=-1)
        self.round = round
    def forward(self , inputs:torch.Tensor):
        inputs = torch.clamp(inputs , 0 , self.A.num_embeddings-1).long()
        ids = torch.clamp(torch.arange(inputs.shape[1] , dtype=torch.long) , 
                          0 , self.U.num_embeddings-1).unsqueeze(0)
        u:torch.Tensor = self.U(ids)
        for _ in range(self.round): 
            a:torch.Tensor = self.A(inputs)
            c:torch.Tensor = self.C(inputs)
            p:torch.Tensor = self.act(torch.bmm(a , u.transpose(2,1)))
            o:torch.Tensor = torch.bmm(p.transpose(1,2) , c)
            u:torch.Tensor = o + u
        ln:torch.Tensor  = self.Lnorms(u)
        out:torch.Tensor = self.out(ln)
        return out
        