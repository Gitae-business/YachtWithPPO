from config import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, actor_critic_model, optimizer, device):
        self.actor_critic = actor_critic_model
        self.optimizer = optimizer
        self.device = device
        
        self.gamma = GAMMA
        self.clip_epsilon = CLIP_EPSILON
        self.gae_lambda = GAE_LAMBDA
        self.ppo_epochs = PPO_EPOCHS
        self.mini_batch_size = MINI_BATCH_SIZE
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []

    def select_action(self, state, available_categories, current_rolls, max_rolls):
        state_tensor = state.unsqueeze(0)
        action_probs, state_value = self.actor_critic(state_tensor)

        if current_rolls >= max_rolls - 1:
            action_probs[0, :32] = 0.0

        for i, is_available in enumerate(available_categories):
            if not is_available:
                action_probs[0, 32 + i] = 0.0
        
        action_probs += 1e-8
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []

    def learn(self, next_state_value):
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)

        # Calculate advantages (GAE)
        returns = []
        advantages = []
        
        # Add the value of the next state for the last step if not done
        # This is crucial for correct GAE calculation
        values_plus_next = torch.cat((values, torch.tensor([next_state_value], dtype=torch.float32).to(self.device)))

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values_plus_next[-1] # Last value is next_state_value
            else:
                next_value = values_plus_next[t+1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * (advantages[-1] if len(advantages) > 0 else 0)
            advantages.append(advantage)
            
            # Calculate returns (targets for value function)
            return_ = rewards[t] + self.gamma * (returns[-1] * (1 - dones[t]) if len(returns) > 0 else next_value * (1 - dones[t]))
            returns.append(return_)

        advantages.reverse()
        returns.reverse()
        
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Optimization Loop
        for _ in range(self.ppo_epochs):
            # Create mini-batches (simple random sampling for now)
            # For more robust implementation, shuffle indices and iterate through batches
            num_samples = len(self.states)
            batch_indices = np.random.choice(num_samples, self.mini_batch_size, replace=False)
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]

            # Get new action probabilities and values
            action_probs, state_values = self.actor_critic(batch_states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            
            # Critic Loss
            critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
            
            # Actor Loss (Clipped Surrogate Objective)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss # 0.5 is a common scaling factor for critic loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.clear_memory()
