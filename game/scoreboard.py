class Scoreboard:
    CATEGORIES = [
        "Aces", "Deuces", "Threes", "Fours", "Fives", "Sixes",
        "Choice", "4 of a Kind", "Full House", "Small Straight",
        "Large Straight", "Yacht"
    ]

    def __init__(self):
        self.scores = {category: None for category in self.CATEGORIES}
        self.reset()

    def reset(self):
        self.scores = {category: None for category in self.CATEGORIES}

    def is_category_used(self, category_name):
        return self.scores[category_name] is not None

    def get_available_categories(self):
        return [cat for cat in self.CATEGORIES if not self.is_category_used(cat)]

    def calculate_score(self, category_name, dice_values):
        sorted_dice = sorted(dice_values)
        counts = {i: sorted_dice.count(i) for i in range(1, 7)}

        if category_name in ["Aces", "Deuces", "Threes", "Fours", "Fives", "Sixes"]:
            target_value = self.CATEGORIES.index(category_name) + 1
            return counts.get(target_value, 0) * target_value
        elif category_name == "Choice":
            return sum(dice_values)
        elif category_name == "4 of a Kind":
            for val, count in counts.items():
                if count >= 4:
                    return sum(dice_values)
            return 0
        elif category_name == "Full House":
            has_three = False
            has_two = False
            for count in counts.values():
                if count == 3:
                    has_three = True
                elif count == 2:
                    has_two = True
            return sum(dice_values) if has_three and has_two else 0
        elif category_name == "Small Straight": # 1-2-3-4, 2-3-4-5, 3-4-5-6
            unique_sorted = sorted(list(set(sorted_dice)))
            if len(unique_sorted) >= 4:
                if (1 in unique_sorted and 2 in unique_sorted and 3 in unique_sorted and 4 in unique_sorted) or \
                   (2 in unique_sorted and 3 in unique_sorted and 4 in unique_sorted and 5 in unique_sorted) or \
                   (3 in unique_sorted and 4 in unique_sorted and 5 in unique_sorted and 6 in unique_sorted):
                    return 30
            return 0
        elif category_name == "Large Straight": # 1-2-3-4-5 or 2-3-4-5-6
            unique_sorted = sorted(list(set(sorted_dice)))
            if len(unique_sorted) == 5:
                if (unique_sorted == [1, 2, 3, 4, 5]) or (unique_sorted == [2, 3, 4, 5, 6]):
                    return 40
            return 0
        elif category_name == "Yacht":
            if len(set(dice_values)) == 1: # All dice are the same
                return 50
            return 0
        else:
            raise ValueError(f"Unknown category: {category_name}")

    def score_category(self, category_name, dice_values):
        if self.is_category_used(category_name):
            raise ValueError(f"Category '{category_name}' already used.")
        
        score = self.calculate_score(category_name, dice_values)
        self.scores[category_name] = score
        return score

    def get_total_score(self):
        total = 0
        upper_section_score = self.get_upper_section_current_score()
        
        if upper_section_score >= 63: # Yacht bonus condition
            total += 35
        
        for score in self.scores.values():
            if score is not None:
                total += score
        
        return total

    def get_upper_section_current_score(self):
        upper_section_score = 0
        for category in self.CATEGORIES[:6]: # Aces to Sixes
            if self.scores[category] is not None:
                upper_section_score += self.scores[category]
        return upper_section_score
