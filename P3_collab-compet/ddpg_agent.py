import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from OUNoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

# Ensure device is consistently used across the whole agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch.optim as optim

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3

class DDPGAgent:
    def __init__(self, state_size, action_size, full_state_size, full_action_size, seed):
        self.actor = Actor(state_size, action_size, seed)
        self.target_actor = Actor(state_size, action_size, seed)
        self.critic = Critic(full_state_size, full_action_size, seed)
        self.target_critic = Critic(full_state_size, full_action_size, seed)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), LR_ACTOR = 1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LR_CRITIC = 1e-3)
        self.noise = lambda: np.random.randn(action_size) * 0.1
     # In DDPGAgent
    def act(self, state, noise=0.0):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu()
        self.actor.train()
        if noise != 0.0:
            action += torch.tensor(self.noise(), dtype=torch.float)
        action = torch.clamp(action, -1, 1)
        return action.squeeze(0)  # return torch.Tensor

    def target_act(self, state, noise=0.0):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(state).cpu()
        self.target_actor.train()
        if noise != 0.0:
            action += torch.tensor(self.noise(), dtype=torch.float)
        action = torch.clamp(action, -1, 1)
        return action.squeeze(0)  # return torch.Tensor


   

