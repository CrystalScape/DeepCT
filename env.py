import gymnasium as  gym 
import torch
from torch import nn
import matplotlib.pyplot as plt
from IPython import display
from DCTNet import DCTNet
import numpy as np
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

env = gym.make('ALE/DemonAttack-v5' , render_mode = 'human')
env.metadata['render_fps'] = 64
act_n = env.action_space.n
models = DCTNet().Models(768 , 12 , 0.5 , 120 , act_n)
optm = torch.optim.RMSprop(models.parameters() , lr=0.01 , alpha=0.9 , eps=1e-5 , 
                           weight_decay=0.00005 , momentum=0.9)
gamma = 0.99
clc = 0.1 

plt.ion()

def Plots(score:list , loss , ret): 
    display.clear_output()
    display.display(plt.gcf())
    plt.clf()
    plt.title('Reward & loss')
    plt.plot(score , label = 'Reward')
    plt.plot(loss , label = 'Loss')
    plt.plot(ret , label = 'Return')
    plt.legend()
    plt.grid()
    plt.show()

print(act_n)
fake_state = env.observation_space
episode = 5
state = env.reset()
state = state[0]
done = False
score = []
losses= []
scores = []
r = []
num_done = 0
while not done : 
    models.train()
    sc = 0
    reward_discount = 0 
    env.render()
    optm.zero_grad()
    state_in = state
    policy , value = models(state_in, state_in)
    print(policy.view(-1))
    act = torch.distributions.Categorical(policy.view(-1)).sample()
    print(act)
    logprob = policy.view(-1)[act]
    state2 , reward , done , _ , _= env.step(act)
    if done :
        #cek point
        num_done += 1
        listdir = os.listdir('m_save')
        if len(listdir) == 0 or len(listdir) < 1: 
            torch.save(models.state_dict() , f'm_save/modelke{num_done}')
        if len(listdir) > 1 : 
            random_c = np.random.choice(listdir)
            dirs = f'm_save/{random_c}'
            models.load_state_dict(torch.load(dirs))
        done = False 
        reward = -10.0
        env.reset()  
    sc += reward
    state = state2
    reward_discount = nn.functional.normalize(
        torch.from_numpy(np.array(reward + gamma*reward_discount)) , dim=0)
    actor_loss = (-1*logprob) * (reward_discount - value.detach())
    critic_loss = torch.pow(value - reward_discount , 2)
    loss = torch.tensor(actor_loss + critic_loss*clc , requires_grad=True)
    loss.backward()
    optm.step()
    score.append(sc)
    print(loss)
    losses.append(loss[0][0].detach().numpy())
    r.append(reward_discount)
    Plots(score ,losses , r)
    print(f"score {sc}")
env.close()