import random
import numpy as np
from collections import deque, namedtuple
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GlobalReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # Each transition stores data for all agents at once
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "obs",           # list of obs per agent [agent][state_dim]
                "obs_full",      # concatenated full observation
                "action",        # list of actions per agent [agent][action_dim]
                "reward",        # list of rewards per agent
                "next_obs",      # list of next obs per agent
                "next_obs_full", # next full observation
                "done"           # list of done flags per agent
            ]
        )

    def add(self, obs, obs_full, action, reward, next_obs, next_obs_full, done):
        """Add a full transition (across all agents) to memory."""
        e = self.experience(obs, obs_full, action, reward, next_obs, next_obs_full, done)
        self.memory.append(e)

    def sample(self):
        """Sample a batch and return transposed lists for each agent."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Transpose: from [batch][agent][value] → [agent][batch][value]
        obs = list(map(list, zip(*[e.obs for e in experiences])))
        next_obs = list(map(list, zip(*[e.next_obs for e in experiences])))
        actions = list(map(list, zip(*[e.action for e in experiences])))
        rewards = list(map(list, zip(*[e.reward for e in experiences])))
        dones = list(map(list, zip(*[e.done for e in experiences])))

        # Convert to tensors per agent
        obs = [torch.tensor(agent, dtype=torch.float32, device=device) for agent in obs]
        next_obs = [torch.tensor(agent, dtype=torch.float32, device=device) for agent in next_obs]
        actions = [torch.tensor(agent, dtype=torch.float32, device=device) for agent in actions]
        rewards = [torch.tensor(agent, dtype=torch.float32, device=device).unsqueeze(1) for agent in rewards]
        dones = [torch.tensor(agent, dtype=torch.float32, device=device).unsqueeze(1) for agent in dones]

        obs_full = torch.tensor([e.obs_full for e in experiences], dtype=torch.float32, device=device)
        next_obs_full = torch.tensor([e.next_obs_full for e in experiences], dtype=torch.float32, device=device)

        return (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)

    def __len__(self):
        return len(self.memory)
