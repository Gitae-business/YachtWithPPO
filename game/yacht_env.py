import numpy as np
from game.dice import DiceSet
from game.scoreboard import Scoreboard
import gymnasium.spaces as spaces

class YachtEnv:
    metadata = {"render_modes": ["human"], "render_fps": 30}
    render_mode = None # Added this line
    MAX_ROLLS = 3 # Max rolls per turn
    NUM_CATEGORIES = len(Scoreboard.CATEGORIES)
    
    ACTION_SPACE_SIZE = 32 + NUM_CATEGORIES # 32 re-roll actions + 12 score actions = 44

    def __init__(self):
        self.dice_set = DiceSet()
        self.scoreboard = Scoreboard()
        self.current_rolls = 0
        self.turn_reward = 0
        self.total_game_score = 0
        self.game_over = False
        
        self.observation_space_shape = (5 + self.NUM_CATEGORIES + 1,)
        self.observation_space = spaces.Box(
            low=np.array([1] * 5 + [0] * self.NUM_CATEGORIES + [0], dtype=np.int32),
            high=np.array([6] * 5 + [1] * self.NUM_CATEGORIES + [self.MAX_ROLLS], dtype=np.int32),
            shape=self.observation_space_shape,
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)
        self.category_names = self.scoreboard.CATEGORIES

    def reset(self):
        self.dice_set.reset()
        self.scoreboard.reset()
        self.current_rolls = 0
        self.turn_reward = 0
        self.total_game_score = 0
        self.game_over = False
        observation = self._get_observation()
        info = {}
        info['available_categories'] = [
            not self.scoreboard.is_category_used(cat) for cat in self.scoreboard.CATEGORIES
        ]
        info['scores_for_categories'] = [
            self.scoreboard.calculate_score(cat, self.dice_set.get_values())
            if not self.scoreboard.is_category_used(cat) else 0
            for cat in self.scoreboard.CATEGORIES
        ]
        return observation, info

    def _get_observation(self):
        dice_obs = np.array(self.dice_set.get_values(), dtype=np.int32)
        
        used_categories_obs = np.array([
            1 if self.scoreboard.is_category_used(cat) else 0
            for cat in self.scoreboard.CATEGORIES
        ], dtype=np.int32)
        
        rolls_obs = np.array([self.current_rolls], dtype=np.int32)
        
        return np.concatenate([dice_obs, used_categories_obs, rolls_obs])

    def step(self, action):
        reward = 0
        done = False
        info = {}

        MAX_SCORE_VALUE = 50.0

        if action < 32: # Re-roll action
            if self.current_rolls < self.MAX_ROLLS - 1:
                current_dice_values = self.dice_set.get_values()
                potential_score_before_reroll = 0
                for cat_name in self.scoreboard.CATEGORIES:
                    if not self.scoreboard.is_category_used(cat_name):
                        potential_score_before_reroll = max(potential_score_before_reroll,
                                                            self.scoreboard.calculate_score(cat_name, current_dice_values))

                reroll_mask = action
                self.dice_set.roll_selected(reroll_mask)
                self.current_rolls += 1
                info['action_type'] = 'reroll'

                new_dice_values = self.dice_set.get_values()
                potential_score_after_reroll = 0
                for cat_name in self.scoreboard.CATEGORIES:
                    if not self.scoreboard.is_category_used(cat_name):
                        potential_score_after_reroll = max(potential_score_after_reroll,
                                                           self.scoreboard.calculate_score(cat_name, new_dice_values))
                
                score_diff = potential_score_after_reroll - potential_score_before_reroll
                
                reward_for_reroll = score_diff / MAX_SCORE_VALUE
                
                if score_diff < 0:
                    reward = reward_for_reroll * 2
                else:
                    reward = reward_for_reroll
                
                if score_diff == 0:
                    reward += 0.05
        else: # Score action
            category_idx = action - 32
            category_name = self.scoreboard.CATEGORIES[category_idx]

            if not self.scoreboard.is_category_used(category_name):
                upper_section_score_before = self.scoreboard.get_upper_section_current_score()
                score_gained = self.scoreboard.score_category(category_name, self.dice_set.get_values())
                
                reward = score_gained / MAX_SCORE_VALUE
                
                if category_name in ['Aces', 'Deuces', 'Threes', 'Fours', 'Fives', 'Sixes']:
                    reward += (score_gained * 0.1) / MAX_SCORE_VALUE
                
                self.total_game_score += score_gained
                self.current_rolls = 0
                self.dice_set.reset()
                info['action_type'] = 'score'
                info['category_scored'] = category_name
                info['score_gained'] = score_gained
                info['total_score'] = self.total_game_score

                upper_section_score_after = self.scoreboard.get_upper_section_current_score()
                if upper_section_score_after >= 63 and upper_section_score_before < 63:
                    reward += 35.0 / MAX_SCORE_VALUE
                    info['bonus_earned'] = True
                else:
                    info['bonus_earned'] = False

                if len(self.scoreboard.get_available_categories()) == 0:
                    done = True
                    reward += self.scoreboard.get_total_score() / MAX_SCORE_VALUE
                    info['final_score'] = self.scoreboard.get_total_score()
        
        observation = self._get_observation()
        if done:
            info['final_score'] = self.scoreboard.get_total_score()

        info['available_categories'] = [not self.scoreboard.is_category_used(cat) for cat in self.scoreboard.CATEGORIES]
        info['scores_for_categories'] = [self.scoreboard.calculate_score(cat, self.dice_set.get_values())
                                         if not self.scoreboard.is_category_used(cat) else 0
                                         for cat in self.scoreboard.CATEGORIES]

        return observation, reward, done, info

    def render(self):
        """Renders the current state of the game (for human viewing)."""
        print("--- Yacht Game State ---")
        print(f"Dice: {self.dice_set.get_values()}")
        print(f"Rolls left: {self.MAX_ROLLS - self.current_rolls}")
        print("Scoreboard:")
        for category, score in self.scoreboard.scores.items():
            status = f"Score: {score}" if score is not None else "Available"
            print(f"  {category}: {status}")
        print(f"Total Score: {self.scoreboard.get_total_score()}")
        print("------------------------")

    def close(self):
        """Clean up any resources (not needed for this simple env)."""
        pass