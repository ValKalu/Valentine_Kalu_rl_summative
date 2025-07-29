# environment/rendering.py
import pygame
import numpy as np
import os

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0) # For correct answers / positive rewards
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0) # For current module/action
RED = (200, 0, 0) # For incorrect answers / negative rewards
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIGHT_GREY = (200, 200, 200)
DARK_GREY = (50, 50, 50)
ACCENT_COLOR = (0, 150, 255) # A nice blue for highlights
PROGRESS_BAR_BG = (70, 70, 70)

class CreativeMindRenderer:
    def __init__(self, domain_names, action_map, render_fps=60):
        pygame.init()
        self.window_width = 1200
        self.window_height = 800
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("CreativeMind Academy RL Environment")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        self.domain_names = domain_names
        self.action_map = action_map
        self.reverse_action_map = {v: k for k, v in action_map.items()}
        self.render_fps = render_fps

        self.domain_icons = self.load_icons()

    def load_icons(self):
        icons = {}
        # Music Business (Briefcase)
        mb_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(mb_surf, ORANGE, (5, 10, 20, 20), 0, 3)
        pygame.draw.rect(mb_surf, ORANGE, (10, 5, 10, 5), 0, 2)
        icons['music_business'] = mb_surf

        # Legal Structures (Gavel)
        ls_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(ls_surf, LIGHT_GREY, (5, 15, 20, 5))
        pygame.draw.circle(ls_surf, LIGHT_GREY, (15, 10), 8)
        pygame.draw.line(ls_surf, BLACK, (15, 10), (15, 25), 2)
        icons['legal_structures'] = ls_surf

        # Digital Marketing (Megaphone)
        dm_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.polygon(dm_surf, PURPLE, [(5,10), (25,5), (25,25), (5,20)])
        pygame.draw.rect(dm_surf, PURPLE, (0,15,10,5))
        icons['digital_marketing'] = dm_surf

        # Music Theory (Music Note)
        mt_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(mt_surf, ACCENT_COLOR, (10, 20), 8)
        pygame.draw.line(mt_surf, ACCENT_COLOR, (18, 20), (18, 5), 3)
        pygame.draw.rect(mt_surf, ACCENT_COLOR, (18, 0, 8, 5))
        icons['music_theory'] = mt_surf
        
        return icons

    def draw_text(self, surface, text, color, x, y, font):
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (x, y))

    def draw_progress_bar(self, surface, x, y, width, height, progress, color):
        pygame.draw.rect(surface, PROGRESS_BAR_BG, (x, y, width, height), 0, 5)
        fill_width = int(width * progress)
        pygame.draw.rect(surface, color, (x, y, fill_width, height), 0, 5)
        pygame.draw.rect(surface, WHITE, (x, y, width, height), 2, 5) # Border

    def render(self, observation, last_action_idx=None, reward_value=None, total_score=0.0, episode_step=0):
        self.screen.fill(DARK_GREY) # Background

        # --- Knowledge Panel ---
        knowledge_panel_x = 20
        knowledge_panel_y = 20
        knowledge_panel_width = 300
        knowledge_panel_height = self.window_height - 40
        pygame.draw.rect(self.screen, BLACK, (knowledge_panel_x, knowledge_panel_y, knowledge_panel_width, knowledge_panel_height), 0, 10)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (knowledge_panel_x, knowledge_panel_y, knowledge_panel_width, knowledge_panel_height), 2, 10)

        self.draw_text(self.screen, "Knowledge Progress", WHITE, knowledge_panel_x + 10, knowledge_panel_y + 10, self.font_large)
        
        # Knowledge bars for each domain
        bar_y_offset = knowledge_panel_y + 60
        bar_width = knowledge_panel_width - 40
        bar_height = 25
        for i, domain in enumerate(self.domain_names):
            knowledge_key = f"knowledge_{domain}"
            if knowledge_key in observation:
                knowledge_val = observation[knowledge_key][0] # Extract scalar from array
                
                icon = self.domain_icons.get(domain)
                if icon:
                    self.screen.blit(icon, (knowledge_panel_x + 10, bar_y_offset + i * 70))

                self.draw_text(self.screen, domain.replace('_', ' ').title(), WHITE, knowledge_panel_x + 50, bar_y_offset + i * 70, self.font_medium)
                
                # Progress bar (normalize knowledge to 0-1 for bar)
                progress = knowledge_val / 100.0 # Assuming max knowledge is 100
                self.draw_progress_bar(self.screen, knowledge_panel_x + 50, bar_y_offset + i * 70 + 25, bar_width - 40, bar_height, progress, GREEN)
                self.draw_text(self.screen, f"{knowledge_val:.1f}%", WHITE, knowledge_panel_x + 50 + (bar_width - 40) // 2 - 20, bar_y_offset + i * 70 + 25 + 2, self.font_small)

        # Total Score
        self.draw_text(self.screen, f"Total Score: {total_score:.0f}", YELLOW, knowledge_panel_x + 10, knowledge_panel_height - 50, self.font_large)
        self.draw_text(self.screen, f"Episode Step: {episode_step}", WHITE, knowledge_panel_x + 10, knowledge_panel_height - 20, self.font_medium)


        # --- Module & Question Panel ---
        main_panel_x = knowledge_panel_x + knowledge_panel_width + 20
        main_panel_y = 20
        main_panel_width = self.window_width - main_panel_x - 20
        main_panel_height = (self.window_height - 60) // 2 # Half height for question
        pygame.draw.rect(self.screen, BLACK, (main_panel_x, main_panel_y, main_panel_width, main_panel_height), 0, 10)
        pygame.draw.rect(self.screen, WHITE, (main_panel_x, main_panel_y, main_panel_width, main_panel_height), 2, 10)
        
        self.draw_text(self.screen, "CreativeMind Academy", WHITE, main_panel_x + 10, main_panel_y + 10, self.font_large)

        # Current Module
        current_module_idx = observation["current_module"]
        current_module_name = self.domain_names[current_module_idx]
        self.draw_text(self.screen, f"Current Module: {current_module_name.replace('_', ' ').title()}", YELLOW, main_panel_x + 20, main_panel_y + 60, self.font_medium)

        # Question Display
        q_difficulty = observation["current_question_difficulty"][0]
        self.draw_text(self.screen, "Question (Difficulty):", WHITE, main_panel_x + 20, main_panel_y + 100, self.font_medium)
        self.draw_text(self.screen, f"{q_difficulty:.2f} (0.0=Easy, 1.0=Hard)", LIGHT_GREY, main_panel_x + 20, main_panel_y + 130, self.font_small)

        # Placeholder for answering
        self.draw_text(self.screen, "Agent is considering its answer...", LIGHT_GREY, main_panel_x + 20, main_panel_y + 180, self.font_medium)


        # --- Feedback Panel ---
        feedback_panel_x = main_panel_x
        feedback_panel_y = main_panel_y + main_panel_height + 20
        feedback_panel_width = main_panel_width
        feedback_panel_height = self.window_height - feedback_panel_y - 20
        pygame.draw.rect(self.screen, BLACK, (feedback_panel_x, feedback_panel_y, feedback_panel_width, feedback_panel_height), 0, 10)
        pygame.draw.rect(self.screen, WHITE, (feedback_panel_x, feedback_panel_y, feedback_panel_width, feedback_panel_height), 2, 10)

        self.draw_text(self.screen, "Agent Feedback", WHITE, feedback_panel_x + 10, feedback_panel_y + 10, self.font_large)

        if last_action_idx is not None:
            action_name = self.reverse_action_map.get(last_action_idx, "Unknown Action")
            self.draw_text(self.screen, f"Last Action: {action_name.replace('_', ' ').title()}", YELLOW, feedback_panel_x + 20, feedback_panel_y + 50, self.font_medium)
        
        if reward_value is not None:
            reward_text_color = GREEN if reward_value >= 0 else RED
            self.draw_text(self.screen, f"Last Reward: {reward_value:.2f}", reward_text_color, feedback_panel_x + 20, feedback_panel_y + 80, self.font_large)
        
        # Display correct/incorrect count
        correct_count = observation.get("questions_answered_correctly", np.array([0]))[0]
        incorrect_count = observation.get("questions_answered_incorrectly", np.array([0]))[0]
        self.draw_text(self.screen, f"Correct: {correct_count}", GREEN, feedback_panel_x + 20, feedback_panel_y + 120, self.font_medium)
        self.draw_text(self.screen, f"Incorrect: {incorrect_count}", RED, feedback_panel_x + 200, feedback_panel_y + 120, self.font_medium)


        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def screen_to_rgb_array(self):
        """Converts the current pygame screen to an RGB numpy array."""
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def close(self):
        pygame.quit()

# Helper function to map action index to a readable string for rendering
def get_action_name(action_index, action_map):
    reverse_map = {v: k for k, v in action_map.items()}
    return reverse_map.get(action_index, f"Unknown Action ({action_index})")

