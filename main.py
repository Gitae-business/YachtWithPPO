from config import *
import torch.optim as optim
from game.yacht_env import YachtEnv
from ppo.model import ActorCritic
from ppo.train import train, load_checkpoint

if __name__ == '__main__':
    env = YachtEnv()

    actor_critic_model = ActorCritic(
        observation_space_shape=env.observation_space_shape,
        action_space_size=env.ACTION_SPACE_SIZE
    )
    optimizer = optim.Adam(actor_critic_model.parameters(), lr=LR)

    # Load checkpoint if exists
    start_episode = 0
    total_timesteps = 0
    start_episode, total_timesteps, current_log_file = load_checkpoint(actor_critic_model, optimizer)

    # 할당된 변수를 train 함수에 전달합니다.
    train(
        actor_critic_model=actor_critic_model,
        optimizer=optimizer,
        start_episode=start_episode,
        total_timesteps=total_timesteps,
        current_log_file=current_log_file
    )
    
