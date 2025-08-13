import os
import csv
import torch
import numpy as np
from config import *
from ppo.ppo_agent import PPOAgent
from game.yacht_env import YachtEnv

def read_last_log_entry(log_file_path):
    if not os.path.exists(log_file_path):
        return 0, 0
    
    with open(log_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            rows = list(reader)
            if len(rows) > 1:
                last_row = rows[-1]
                episode = int(last_row[0])
                total_timesteps = int(last_row[3])
                return episode, total_timesteps
        except (IOError, ValueError, IndexError) as e:
            print(f"Error reading log file: {e}. Starting from scratch.")
            return 0, 0
    return 0, 0

def save_checkpoint(actor_critic_model, optimizer, episode, total_timesteps, log_file_path):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    torch.save({
        'model_state_dict': actor_critic_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'total_timesteps': total_timesteps,
        'log_file_path': log_file_path
    }, checkpoint_path)
    print(f"Latest checkpoint saved to {checkpoint_path}")

def load_checkpoint(actor_critic_model, optimizer):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        actor_critic_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_episode = checkpoint.get('episode', 0)
        total_timesteps = checkpoint.get('total_timesteps', 0)
        log_file_path = checkpoint.get('log_file_path', LOG_FILE)

        print(f"Resuming training from Episode {start_episode}, Total Timesteps {total_timesteps}.")
        return start_episode, total_timesteps, log_file_path
    else:
        print("No latest checkpoint found. Starting new training.")
        return 0, 0, LOG_FILE

def train(actor_critic_model, optimizer, start_episode, total_timesteps, current_log_file):
    device_ppo = torch.device("cpu")
    env = YachtEnv()

    agent = PPOAgent(
        actor_critic_model=actor_critic_model,
        optimizer=optimizer,
        device=device_ppo
    )

    episode_rewards = []
    episode_final_scores = []

    write_header = not os.path.exists(current_log_file) or os.stat(current_log_file).st_size == 0

    try:
        csv_file = open(current_log_file, 'a', newline='')
        csv_writer = csv.writer(csv_file)

        if write_header:
            csv_writer.writerow(['Episode', 'Avg_Reward_100', 'Avg_Final_Score_100', 'Total_Timesteps'])
            csv_file.flush()
        
        last_logged_episode, last_logged_timesteps = read_last_log_entry(current_log_file)
        start_episode = max(start_episode, last_logged_episode)
        total_timesteps = max(total_timesteps, last_logged_timesteps)

        print(f"Starting PPO training from Episode {start_episode}, Total Timesteps {total_timesteps}...")

        for episode in range(start_episode, NUM_EPISODES):
            state, info = env.reset()
            state = torch.FloatTensor(state).to(device_ppo)
            done = False
            episode_reward = 0
            steps_in_episode = 0

            while not done and steps_in_episode < MAX_STEPS_PER_EPISODE:
                available_categories = info['available_categories']
                current_rolls = env.current_rolls
                
                action, log_prob, value = agent.select_action(state, available_categories, current_rolls, env.MAX_ROLLS)
                
                next_state, reward, done, info = env.step(action)
                next_state = torch.FloatTensor(next_state).to(device_ppo)

                agent.store_transition(state, action, reward, log_prob, value, done)
                
                state = next_state
                episode_reward += reward
                total_timesteps += 1
                steps_in_episode += 1

                if total_timesteps % UPDATE_TIMESTEPS == 0:
                    _, next_value = agent.actor_critic(next_state.unsqueeze(0))
                    agent.learn(next_value.item())

            episode_rewards.append(episode_reward)
            final_score = info.get('final_score', env.scoreboard.get_total_score())
            episode_final_scores.append(final_score)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_final_score = np.mean(episode_final_scores[-100:])
                print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Final Score: {avg_final_score:.2f} | Total Timesteps: {total_timesteps}")
                csv_writer.writerow([episode + 1, f"{avg_reward:.2f}", f"{avg_final_score:.2f}", total_timesteps])
                csv_file.flush()

            if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(actor_critic_model, optimizer, episode + 1, total_timesteps, current_log_file)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint and closing log file...")
    finally:
        if 'csv_file' in locals() and not csv_file.closed:
            csv_file.close()
        print("Training finished. Log file closed.")

        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_checkpoint.pth")
        torch.save({
            'model_state_dict': actor_critic_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode + 1,
            'total_timesteps': total_timesteps,
            'log_file_path': current_log_file
        }, final_checkpoint_path)
        print(f"Final checkpoint saved to {final_checkpoint_path}")