import torch
import numpy as np

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*source_param.data + (1-tau)*target_param.data)

def transpose_to_tensor(batch):
    # batch: list of lists (agents) of arrays (batch_size)
    return [torch.tensor(np.stack(x), dtype=torch.float) for x in zip(*batch)]
