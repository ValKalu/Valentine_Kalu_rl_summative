import pygame
import numpy as np
import os

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0)
RED = (200, 0, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIGHT_GREY = (200, 200, 200)
DARK_GREY = (50, 50, 50)
ACCENT_COLOR = (0, 150, 255) # A nice blue for highlights

class TalentForgeRenderer:
    def __init__(self, creative_profile_keys, action_map, render_fps=60):
        pygame.init()
        self.window_width = 1200
        self.window_height = 800
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("TalentForge Connect Environment")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        self.creative_profile_keys = creative_profile_keys
        self.action_map = action_map
        self.reverse_action_map = {v: k for k, v in action_map.items()}
        self.render_fps = render_fps

        # Load or create creative avatar
        self.creative_avatar = self.load_creative_avatar()
        self.opportunity_icons = self.load_opportunity_icons()

    def load_creative_avatar(self):
        # Create a simple SVG-like avatar using Pygame drawing
        avatar_surface = pygame.Surface((100, 100), pygame.SRCALPHA) # SRCALPHA for transparency
        pygame.draw.circle(avatar_surface, ACCENT_COLOR, (50, 50), 45)
        pygame.draw.circle(avatar_surface, WHITE, (50, 50), 40)
        pygame.draw.circle(avatar_surface, BLUE, (50, 40), 10) # Eye
        pygame.draw.circle(avatar_surface, BLUE, (70, 40), 10) # Eye
        pygame.draw.arc(avatar_surface, RED, (40, 60, 40, 20), 0, np.pi, 2) # Mouth
        return avatar_surface

    def load_opportunity_icons(self):
        # Create simple icons for opportunities
        icons = {}
        # Education icon (book)
        edu_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(edu_surf, YELLOW, (5, 0, 20, 30), 0, 5)
        pygame.draw.rect(edu_surf, ORANGE, (0, 5, 30, 20), 0, 5)
        icons['education'] = edu_surf

        # Gig icon (microphone)
        gig_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(gig_surf, LIGHT_GREY, (15, 10), 10)
        pygame.draw.rect(gig_surf, LIGHT_GREY, (10, 20, 10, 10))
        pygame.draw.line(gig_surf, BLACK, (15, 30), (15, 20), 2)
        icons['gigs'] = gig_surf

        # Brand icon (star)
        brand_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.polygon(brand_surf, PURPLE, [(15, 0), (20, 10), (30, 10), (22, 18), (25, 28), (15, 22), (5, 28), (8, 18), (0, 10), (10, 10)])
        icons['brands'] = brand_surf

        # Investor icon (money bag)
        inv_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(inv_surf, GREEN, (5, 10, 20, 20), 0, 5)
        pygame.draw.circle(inv_surf, GREEN, (15, 10), 10)
        pygame.draw.line(inv_surf, WHITE, (10, 15), (20, 15), 2)
        icons['investors'] = inv_surf
        
        return icons

    def draw_text(self, surface, text, color, x, y, font):
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (x, y))

    def render(self, observation, last_action_idx=None, reward_value=None, total_reward=0.0, episode_step=0):
        self.screen.fill(DARK_GREY) # Background

        # --- Creative Profile Panel ---
        profile_panel_x = 20
        profile_panel_y = 20
        profile_panel_width = 300
        profile_panel_height = self.window_height - 40
        pygame.draw.rect(self.screen, BLACK, (profile_panel_x, profile_panel_y, profile_panel_width, profile_panel_height), 0, 10)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (profile_panel_x, profile_panel_y, profile_panel_width, profile_panel_height), 2, 10)

        self.draw_text(self.screen, "Creative Profile", WHITE, profile_panel_x + 10, profile_panel_y + 10, self.font_large)
        
        # Avatar
        avatar_x = profile_panel_x + (profile_panel_width - self.creative_avatar.get_width()) // 2
        avatar_y = profile_panel_y + 50
        self.screen.blit(self.creative_avatar, (avatar_x, avatar_y))

        # Profile Stats
        stat_y_offset = avatar_y + self.creative_avatar.get_height() + 20
        for i, key in enumerate(self.creative_profile_keys):
            if key in observation:
                value = observation[key]
                # Handle numpy arrays of shape (1,) by extracting the scalar
                display_value = value[0] if isinstance(value, np.ndarray) and value.size == 1 else value
                
                # Special handling for genre and status
                if key == "genre_specialization":
                    genre_map = {0: "HipHop", 1: "Afrobeat", 2: "R&B", 3: "Gospel", 4: "Other"}
                    display_value = genre_map.get(display_value, "Unknown")
                elif key == "current_engagement_status":
                    status_map = {0: "Idle", 1: "Training", 2: "Performing", 3: "Collaborating"}
                    display_value = status_map.get(display_value, "Unknown")

                text = f"{key.replace('_', ' ').title()}: {display_value:.2f}" if isinstance(display_value, float) else f"{key.replace('_', ' ').title()}: {display_value}"
                self.draw_text(self.screen, text, LIGHT_GREY, profile_panel_x + 10, stat_y_offset + i * 25, self.font_medium)

        # Total Earnings
        self.draw_text(self.screen, f"Total Earnings: ${total_reward:.2f}", YELLOW, profile_panel_x + 10, profile_panel_height - 50, self.font_large)
        self.draw_text(self.screen, f"Episode Step: {episode_step}", WHITE, profile_panel_x + 10, profile_panel_height - 20, self.font_medium)


        # --- Opportunity Dashboard Panel ---
        dashboard_panel_x = profile_panel_x + profile_panel_width + 20
        dashboard_panel_y = 20
        dashboard_panel_width = self.window_width - dashboard_panel_x - 20
        dashboard_panel_height = (self.window_height - 60) // 2 # Half height for opportunities
        pygame.draw.rect(self.screen, BLACK, (dashboard_panel_x, dashboard_panel_y, dashboard_panel_width, dashboard_panel_height), 0, 10)
        pygame.draw.rect(self.screen, WHITE, (dashboard_panel_x, dashboard_panel_y, dashboard_panel_width, dashboard_panel_height), 2, 10)
        
        self.draw_text(self.screen, "Available Opportunities", WHITE, dashboard_panel_x + 10, dashboard_panel_y + 10, self.font_large)

        opp_display_y_offset = dashboard_panel_y + 50
        opportunity_info = {
            "education": {"slots": observation["available_education_slots"][0], "quality": observation["opportunity_quality_education"][0]},
            "gigs": {"count": observation["available_gigs"][0], "quality": observation["opportunity_quality_gig"][0]},
            "brands": {"count": observation["available_brand_deals"][0], "quality": observation["opportunity_quality_brand"][0]},
            "investors": {"count": observation["available_investor_meetings"][0], "quality": observation["opportunity_quality_investor"][0]}
        }

        for i, (opp_type, opp_data) in enumerate(opportunity_info.items()):
            icon = self.opportunity_icons.get(opp_type)
            if icon:
                self.screen.blit(icon, (dashboard_panel_x + 20, opp_display_y_offset + i * 60))
            
            self.draw_text(self.screen, opp_type.title(), WHITE, dashboard_panel_x + 60, opp_display_y_offset + i * 60, self.font_medium)
            
            count_key = "slots" if opp_type == "education" else "count"
            self.draw_text(self.screen, f"Available: {opp_data[count_key]}", LIGHT_GREY, dashboard_panel_x + 60, opp_display_y_offset + i * 60 + 20, self.font_small)
            self.draw_text(self.screen, f"Quality: {opp_data['quality']:.2f}", LIGHT_GREY, dashboard_panel_x + 180, opp_display_y_offset + i * 60 + 20, self.font_small)

        # --- Action & Reward Feedback Panel ---
        feedback_panel_x = dashboard_panel_x
        feedback_panel_y = dashboard_panel_y + dashboard_panel_height + 20
        feedback_panel_width = dashboard_panel_width
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
