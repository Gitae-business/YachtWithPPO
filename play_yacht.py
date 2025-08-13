import gymnasium as gym
import numpy as np
from game.yacht_env import YachtEnv

def play_human_game():
    env = YachtEnv()
    
    print("Welcome to Yacht!")
    print("You will play against yourself to get the highest score.")
    print("The game consists of 12 turns. In each turn, you roll 5 dice.")
    print("You can re-roll up to 2 times per turn. Then you choose a category to score.")
    print("Let's begin!")

    while True:
        obs, info = env.reset()
        done = False
        truncated = False
        total_score = 0
        turn = 0

        while not done and not truncated:
            turn += 1
            print(f"\n--- Turn {turn}/12 ---")
            
            current_dice = obs[0:5] # Extract dice values from the observation array
            reroll_count = 0

            for roll_num in range(3):
                env.render()
                print(f"Roll {roll_num + 1}/3. Current dice: {current_dice}")

                if roll_num < 2: # Allow up to 2 re-rolls
                    while True:
                        try:
                            reroll_input = input("Enter dice indices to re-roll (1-5, comma-separated), or press Enter to keep all: ").strip()
                            if not reroll_input:
                                break
                            
                            indices_to_reroll = [int(i) - 1 for i in reroll_input.split(',')]
                            if not all(0 <= i <= 4 for i in indices_to_reroll):
                                raise ValueError("Indices must be between 1 and 5.")
                            
                            # Action for re-roll: a boolean array indicating which dice to re-roll
                            reroll_action = np.zeros(5, dtype=bool)
                            for idx in indices_to_reroll:
                                reroll_action[idx] = True
                            
                            # The action space for re-rolling is 2^5 = 32 options.
                            # Convert boolean array to integer action.
                            action_value = 0
                            for i in range(5):
                                if reroll_action[i]:
                                    action_value += (1 << i) # Bitwise operation to convert boolean array to integer
                            
                            # The first 32 actions are for re-rolling
                            obs, reward, done, info = env.step(action_value)
                            current_dice = obs[0:5] # Extract dice values from the observation array
                            break
                        except ValueError as e:
                            print(f"Invalid input: {e}. Please enter comma-separated numbers between 0 and 4.")
                        except Exception as e:
                            print(f"An error occurred: {e}. Please try again.")
                else:
                    print("No more re-rolls available for this turn.")
            
            env.render() # Final dice state after re-rolls
            print(f"Final dice for this turn: {current_dice}")
            print("Available categories and their scores:")
            
            # Display available categories and their potential scores
            available_categories = info['available_categories']
            scores_for_categories = info['scores_for_categories']

            for i in range(env.action_space.n - 32): # Iterate through scoring categories
                if available_categories[i]:
                    print(f"  {i + 1}: {env.category_names[i]} (Score: {scores_for_categories[i]})")
                else:
                    print(f"  {i + 1}: {env.category_names[i]} (Already scored)")

            while True:
                try:
                    category_input = input("Enter category to score (number): ").strip()
                    category_idx = int(category_input) - 1
                    
                    if not (0 <= category_idx < (env.action_space.n - 32)):
                        raise ValueError(f"Category index must be between 1 and {env.action_space.n - 32}.")
                    
                    if not available_categories[category_idx]:
                        print("This category has already been scored. Please choose an available one.")
                        continue
                    
                    # Action for scoring: offset by 32 because first 32 actions are re-rolls
                    action_value = 32 + category_idx
                    obs, reward, done, info = env.step(action_value)
                    total_score = info['total_score']
                    print(f"Scored {env.category_names[category_idx]} for {reward} points. Total score: {total_score}")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please enter a valid category number.")
                except Exception as e:
                    print(f"An error occurred: {e}. Please try again.")

        env.close()
        print(f"\n--- Game Over! ---")
        print(f"Final Score: {total_score}")

        play_again = input("Play again? (yes/no): ").strip().lower()
        if play_again != 'yes':
            break

if __name__ == "__main__":
    play_human_game()
