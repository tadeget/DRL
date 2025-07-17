import torch
from utilities import soft_update, transpose_to_tensor
import numpy as np

class MADDPG:
    def __init__(self, agents, discount_factor=0.95, tau=0.02):
        self.maddpg_agent = agents
        self.discount_factor = discount_factor
        self.tau = tau

    def act(self, obs_all_agents, noise=0.0):
        return [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]

    def target_act(self, obs_all_agents, noise=0.0):
        return [agent.target_act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]

    def update(self, samples, agent_number):
        #obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        #obs_full = torch.cat(obs_full, dim=1)
        #next_obs_full = torch.cat(next_obs_full, dim=1)
        
        #obs_full = obs_full[0]  # it's already a tensor of shape [batch_size, full_state_size]
        #next_obs_full = next_obs_full[0]
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        obs_full = torch.tensor(np.stack(obs_full), dtype=torch.float)
        next_obs_full = torch.tensor(np.stack(next_obs_full), dtype=torch.float)
        # For obs, action, next_obs, use transpose_to_tensor if needed
        obs = transpose_to_tensor(obs)
        action = transpose_to_tensor(action)
        reward = transpose_to_tensor(reward)
        next_obs = transpose_to_tensor(next_obs)
        done = transpose_to_tensor(done)


        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # --- Critic Update ---
        #print("next_obs_full shape:", next_obs_full.shape)  # Should be [batch_size, full_state_size]
        target_actions = self.target_act(next_obs)
        #xprint([a.shape for a in target_actions])  # Each should be [batch_size, action_size]
        target_actions = torch.cat(target_actions, dim=1)
        #target_critic_input = torch.cat((next_obs_full, target_actions), dim=1)

        #with torch.no_grad():
        #    q_next = agent.target_critic(target_critic_input)
        
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full, target_actions)

        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action_cat = torch.cat(action, dim=1)
        #critic_input = torch.cat((obs_full, action_cat), dim=1)
        #q = agent.critic(critic_input)
        q = agent.critic(obs_full, action_cat)
        critic_loss = torch.nn.functional.mse_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        # --- Actor Update ---
        agent.actor_optimizer.zero_grad()
        q_input = [
            self.maddpg_agent[i].actor(ob) if i == agent_number
            else self.maddpg_agent[i].actor(ob).detach()
            for i, ob in enumerate(obs)
        ]
        q_input = torch.cat(q_input, dim=1)
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self):
        for agent in self.maddpg_agent:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)
