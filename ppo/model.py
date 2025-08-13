import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, observation_space_shape, action_space_size):
        super(ActorCritic, self).__init__()
        
        self.fc1 = nn.Linear(observation_space_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        self.actor_head = nn.Linear(256, action_space_size)
        
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)
        
        return action_probs, state_value
