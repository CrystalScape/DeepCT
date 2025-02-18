import torch 
dum = torch.randn((1, 5, 768))

def Scale_Dot_Product(q:torch.Tensor , k:torch.Tensor , v:torch.Tensor , mask:bool): 
    dimK = torch.sqrt(torch.tensor(k.shape[-1]))
    attention:torch.Tensor = torch.bmm(q , k.transpose(1,2)) / dimK
    if mask : 
        masking = torch.tril(torch.ones(attention.shape[-1] , 
                                        attention.shape[-1])).unsqueeze(0)
        attention = attention.masked_fill(masking == 0 , -float('inf'))
        attention = torch.nn.functional.softmax(attention , -1)
        return torch.bmm(attention , v)
    else :
        attention = torch.nn.functional.softmax(attention , -1)
        return torch.bmm(attention , v)

class HeadAttention(torch.nn.Module):
    def __init__(self , dim_head , num_hidden , mask:bool):
        super(HeadAttention , self).__init__()
        self.Lin1 = torch.nn.Linear(num_hidden , dim_head)
        self.Lin2 = torch.nn.Linear(num_hidden , dim_head)
        self.Lin3 = torch.nn.Linear(num_hidden , dim_head)
        self.mask = mask
    def forward(self  , inputs): 
        sdp = Scale_Dot_Product(self.Lin1(inputs) , self.Lin2(inputs) , 
                                 self.Lin3(inputs) , mask=self.mask)
        return sdp

class Attentions(torch.nn.Module): 
    def __init__(self , hidden_size , num_head , mask:bool):
        super(Attentions , self).__init__()
        self.dim_head = hidden_size // num_head
        self.out = torch.nn.Linear(hidden_size , hidden_size)
        self.heads = torch.nn.ModuleList([
            HeadAttention(self.dim_head , hidden_size , mask) for _ in range(num_head)
        ])
    def forward(self , inputs): 
        x = torch.cat([h(inputs) for h in self.heads] , -1)
        x = self.out(x)
        return x
    
class FeedForward(torch.nn.Module): 
    def __init__(self , hidden_size , drop_size):
        super(FeedForward , self).__init__()
        self.l1 = torch.nn.Linear(hidden_size , hidden_size)
        self.l2 = torch.nn.Linear(hidden_size , hidden_size)
        self.gel = torch.nn.GELU()
        self.drop = torch.nn.Dropout(drop_size)
    def forward(self , inputs): 
        x = self.l1(inputs)
        x = self.gel(x)
        x = self.l2(x)
        x = self.drop(x)
        return x 

class EncoderLayers(torch.nn.Module): 
    def __init__(self , hidden_size , num_head, drop_rate):
        super(EncoderLayers , self).__init__()
        self.lnorms = torch.nn.LayerNorm(hidden_size)
        self.att = Attentions(hidden_size , num_head , mask = False)
        self.ffn = FeedForward(hidden_size , drop_rate)
    def forward(self , inputs): 
        x = inputs + self.att(self.lnorms(inputs))
        x = x + self.ffn(self.lnorms(x))
        return x 
    
class DecoderLayers(torch.nn.Module): 
    def __init__(self , hidden_size , num_head, drop_rate):
        super(DecoderLayers , self).__init__()
        self.lnorms = torch.nn.LayerNorm(hidden_size)
        self.att1 = Attentions(hidden_size , num_head , mask=True)
        self.att2 = Attentions(hidden_size , num_head , mask=False)
        self.ffn = FeedForward(hidden_size , drop_rate)
    def forward(self , inputs): 
        x = inputs + self.att1(self.lnorms(inputs))
        x = x + self.att2(self.lnorms(x))
        x = x + self.ffn(self.lnorms(x))
        return x        
