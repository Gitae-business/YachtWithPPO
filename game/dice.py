import random

class Dice:
    def __init__(self):
        self.value = 0
        self.roll()

    def roll(self):
        self.value = random.randint(1, 6)
        return self.value

class DiceSet:
    def __init__(self):
        self.dice = [Dice() for _ in range(5)]
        self.reset()

    def reset(self):
        for die in self.dice:
            die.roll()

    def get_values(self):
        return [die.value for die in self.dice]

    def roll_selected(self, reroll_mask):
        """
        Re-rolls selected dice based on a bitmask.
        reroll_mask: An integer where each bit represents a die (0-4).
                     If the bit is set, the corresponding die is re-rolled.
                     e.g., 0b00001 (1) re-rolls the first die.
                           0b10000 (16) re-rolls the last die.
                           0b11111 (31) re-rolls all dice.
        """
        for i in range(5):
            if (reroll_mask >> i) & 1:
                self.dice[i].roll()