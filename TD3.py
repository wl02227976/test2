#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PER import PER


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic1(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic1, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)



	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class Critic2(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic2, self).__init__()


		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)


		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q2


	def Q2(self, state, action):
		sa = torch.cat([state, action], 1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q2


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic1 = Critic1(state_dim, action_dim).to(device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)

		self.critic2 = Critic2(state_dim, action_dim).to(device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
        


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
    
    
	def step(self, state, action, reward, next_state, done):
		"""Save experience in replay memory."""
		# Set reward as initial priority, see:
		#   https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
		##print('test')
		replay_buffer.add((state, action, reward, next_state, done), reward)
        
        
	def mse(self, expected, targets, is_weights):
		"""Custom loss function that takes into account the importance-sampling weights."""
		td_error = expected - targets
		weighted_squared_error = is_weights * td_error * td_error
		return torch.sum(weighted_squared_error) / torch.numel(weighted_squared_error)


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		##state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		idxs, experiences, is_weights = replay_buffer.sample(batch_size)
        
		##print('test')
		##print(experiences)
		##state = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
		##print('test')
		##action = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
		##reward = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
		##next_state = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
		##not_done = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
		state = torch.from_numpy(np.vstack([e[0] for e in experiences if ((e !=0) and (e is not None))])).float().to(device)
		action = torch.from_numpy(np.vstack([e[1] for e in experiences if ((e !=0) and (e is not None))])).float().to(device)
		reward = torch.from_numpy(np.vstack([e[2] for e in experiences if ((e !=0) and (e is not None))])).float().to(device)
		next_state = torch.from_numpy(np.vstack([e[3] for e in experiences if ((e !=0) and (e is not None))])).float().to(device)
		not_done = torch.from_numpy(np.vstack([e[4] for e in experiences if ((e !=0) and (e is not None))]).astype(np.uint8)).float().to(device)
        
		##print(action)
		##print(experiences[0])
            
		is_weights =  torch.from_numpy(is_weights).float().to(device)
        

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1 = self.critic1_target(next_state, next_action)
			target_Q2 = self.critic2_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1 = self.critic1(state, action)
		errors1 = np.abs((current_Q1 - target_Q).detach().cpu().numpy())        
		critic1_loss = self.mse(current_Q1, target_Q, is_weights)
		self.critic1_optimizer.zero_grad()
		critic1_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1)
		self.critic1_optimizer.step()
        
		replay_buffer.batch_update(idxs, errors1)
        
		current_Q2 = self.critic2(state, action)
		critic2_loss = self.mse(current_Q2, target_Q, is_weights)
		self.critic2_optimizer.zero_grad()
		critic2_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1)
		self.critic2_optimizer.step()




		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
			for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic1.state_dict(), filename + "_critic1")
		torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        
		torch.save(self.critic2.state_dict(), filename + "_critic2")
		torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic1.load_state_dict(torch.load(filename + "_critic1"))
		self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
		self.critic1_target = copy.deepcopy(self.critic1)
        
		self.critic2.load_state_dict(torch.load(filename + "_critic2"))
		self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
		self.critic2_target = copy.deepcopy(self.critic2)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		

