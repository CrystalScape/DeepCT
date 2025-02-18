import torch 
import numpy as np 
import pandas as pd

class Imagine : 
    
    def __init__(self , state , reward):
        states = list()
        states.append(state)
        rewards = list()
        rewards.append(reward)
        