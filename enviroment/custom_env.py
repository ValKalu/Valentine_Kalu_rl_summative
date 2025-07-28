import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.rendering import TalentForgeRenderer # Import your renderer

class TalentForgeConnectEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Define Action Space
        # Map actions to integers for Discrete space
        self.ACTION_MAP = {
            "enroll_education_music_prod": 0,
            "enroll_education_vocal": 1,
            "apply_gig_local": 2,
            "apply_gig_regional": 3,
            "propose_brand_basic": 4,
            "schedule_investor_seed": 5,
            "focus_portfolio_improvement": 6,
            "network_online": 7,
            "rest_idle": 8,
            "explore_new_opportunities": 9 # Added an action to refresh opportunities
        }
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.reverse_action_map = {v: k for k, v in self.ACTION_MAP.items()} # For rendering

        # Define Observation Space (Using a dictionary for structured observation)
        self.observation_space = spaces.Dict({
            # Creative's Profile (all Box spaces need shape=(1,) for single values)
            "skill_level": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32),
            "experience_points": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "portfolio_quality_score": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "genre_specialization": spaces.Discrete(5), # 0:HipHop, 1:Afrobeat, 2:R&B, 3:Gospel, 4:Other
            "current_engagement_status": spaces.Discrete(4), # 0:Idle, 1:Training, 2:Performing, 3:Collaborating
            "social_media_reach": spaces.Box(100.0, 1_000_000.0, shape=(1,), dtype=np.float32), # Min 100, Max 1M
            "monetary_earnings_cumulative": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "time_since_last_engagement": spaces.Box(0, 365, shape=(1,), dtype=np.int32), # Days since last active engagement

            # Available Opportunities (Dynamic)
            "available_education_slots": spaces.Box(0, 5, shape=(1,), dtype=np.int32),
            "available_gigs": spaces.Box(0, 10, shape=(1,), dtype=np.int32),
            "available_brand_deals": spaces.Box(0, 3, shape=(1,), dtype=np.int32),
            "available_investor_meetings": spaces.Box(0, 2, shape=(1,), dtype=np.int32),
            # Quality scores for current opportunities (0.0 to 1.0)
            "opportunity_quality_education": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "opportunity_quality_gig": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "opportunity_quality_brand": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "opportunity_quality_investor": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        # Environment State Variables (Internal, will be reset at each episode)
        self.creative_profile = {}
        self.available_opportunities = {}
        self.current_step = 0
        self.max_steps = 365 * 3 # Simulate 3 years for an episode, one step per day

        # Rendering setup
        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode in ["human", "rgb_array"]:
            # Pass the keys for the creative profile section and action map to the renderer
            self.renderer = TalentForgeRenderer(list(self.observation_space.spaces.keys())[:8], self.ACTION_MAP, self.metadata["render_fps"])

        self.total_episode_reward = 0.0 # Track total reward for rendering

    def _get_obs(self):
        # Combine creative profile and available opportunities into one dictionary for observation
        obs = {}
        # Creative Profile values (convert to numpy arrays of shape (1,) if they are scalars)
        # Ensure keys match those defined in observation_space
        for k, v in self.creative_profile.items():
            if k in ["skill_level", "experience_points", "portfolio_quality_score",
                     "social_media_reach", "monetary_earnings_cumulative", "time_since_last_engagement"]:
                obs[k] = np.array([v], dtype=self.observation_space.spaces[k].dtype)
            else: # For discrete values like genre or status
                obs[k] = v 

        # Opportunity values (convert to numpy arrays of shape (1,))
        obs["available_education_slots"] = np.array([self.available_opportunities["education"]["slots"]], dtype=np.int32)
        obs["available_gigs"] = np.array([self.available_opportunities["gigs"]["count"]], dtype=np.int32)
        obs["available_brand_deals"] = np.array([self.available_opportunities["brands"]["count"]], dtype=np.int32)
        obs["available_investor_meetings"] = np.array([self.available_opportunities["investors"]["count"]], dtype=np.int32)
        obs["opportunity_quality_education"] = np.array([self.available_opportunities["education"]["quality"]], dtype=np.float32)
        obs["opportunity_quality_gig"] = np.array([self.available_opportunities["gigs"]["quality"]], dtype=np.float32)
        obs["opportunity_quality_brand"] = np.array([self.available_opportunities["brands"]["quality"]], dtype=np.float32)
        obs["opportunity_quality_investor"] = np.array([self.available_opportunities["investors"]["quality"]], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "monetary_earnings": self.creative_profile["monetary_earnings_cumulative"],
            "skill_level": self.creative_profile["skill_level"],
            "social_media_reach": self.creative_profile["social_media_reach"]
        }

    def _update_opportunities(self):
        """Simulates dynamic changes in available opportunities."""
        # Opportunities are more likely to appear if creative is doing well
        base_quality_factor = (self.creative_profile["portfolio_quality_score"] + self.creative_profile["skill_level"] / 10.0) / 2.0
        
        # Education: Always available, but quality varies
        self.available_opportunities["education"]["slots"] = random.randint(1, 4)
        self.available_opportunities["education"]["quality"] = min(1.0, random.uniform(0.4, 0.8) + base_quality_factor * 0.1)

        # Gigs: More gigs if skill/reach is higher
        self.available_opportunities["gigs"]["count"] = random.randint(1, 7) if self.creative_profile["skill_level"] > 2 else random.randint(0, 3)
        self.available_opportunities["gigs"]["quality"] = min(1.0, random.uniform(0.3, 0.7) + base_quality_factor * 0.2)

        # Brands: Appear if social media reach is good
        self.available_opportunities["brands"]["count"] = 1 if self.creative_profile["social_media_reach"] > 5000 and random.random() < 0.5 else 0
        self.available_opportunities["brands"]["quality"] = min(1.0, random.uniform(0.6, 0.9) + base_quality_factor * 0.3)

        # Investors: Only appear if creative has significant earnings and portfolio
        self.available_opportunities["investors"]["count"] = 1 if self.creative_profile["monetary_earnings_cumulative"] > 10000 and self.creative_profile["portfolio_quality_score"] > 0.8 and random.random() < 0.3 else 0
        self.available_opportunities["investors"]["quality"] = min(1.0, random.uniform(0.7, 1.0) + base_quality_factor * 0.4)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset creative profile to initial state
        self.creative_profile = {
            "skill_level": 1.0,
            "experience_points": 0.0,
            "portfolio_quality_score": 0.1,
            "genre_specialization": random.randint(0, 4), # Random initial genre
            "current_engagement_status": 0, # Idle
            "social_media_reach": 100.0,
            "monetary_earnings_cumulative": 0.0,
            "time_since_last_engagement": 0
        }

        # Initialize opportunities
        self.available_opportunities = {
            "education": {"slots": 0, "quality": 0.0},
            "gigs": {"count": 0, "quality": 0.0},
            "brands": {"count": 0, "quality": 0.0},
            "investors": {"count": 0, "quality": 0.0}
        }
        self._update_opportunities() # Populate initial opportunities

        self.current_step = 0
        self.last_action_idx = None
        self.last_reward_value = None
        self.total_episode_reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.renderer.render(observation, None, None, self.total_episode_reward, self.current_step)

        return observation, info

    def step(self, action):
        reward = -0.1 # Small base time penalty to encourage efficiency
        done = False
        self.current_step += 1
        self.creative_profile["time_since_last_engagement"] += 1 # Increment time since last engagement

        action_name = self.reverse_action_map[action]
        self.last_action_idx = action # Store for rendering

        # --- Implement Action Effects and Rewards ---
        # Success chance is influenced by creative's stats and opportunity quality
        current_skill = self.creative_profile["skill_level"]
        current_portfolio = self.creative_profile["portfolio_quality_score"]
        current_reach = self.creative_profile["social_media_reach"]
        current_earnings = self.creative_profile["monetary_earnings_cumulative"]

        if action_name == "enroll_education_music_prod":
            if self.available_opportunities["education"]["slots"] > 0:
                base_success_chance = self.available_opportunities["education"]["quality"]
                success_chance = min(0.95, base_success_chance + (current_skill / 20.0)) # Higher skill, better chance
                if random.random() < success_chance:
                    self.creative_profile["skill_level"] = min(10.0, current_skill + random.uniform(0.2, 0.5))
                    self.creative_profile["portfolio_quality_score"] = min(1.0, current_portfolio + random.uniform(0.02, 0.08))
                    self.creative_profile["experience_points"] += 15
                    reward += 20.0 # Positive reward for education
                    self.available_opportunities["education"]["slots"] -= 1
                    self.creative_profile["current_engagement_status"] = 1 # In_Training
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 5.0 # Penalty for failed enrollment
            else:
                reward -= 2.0 # Penalty for trying to enroll with no slots

        elif action_name == "enroll_education_vocal":
            if self.available_opportunities["education"]["slots"] > 0: # Assuming same slot pool
                base_success_chance = self.available_opportunities["education"]["quality"]
                success_chance = min(0.95, base_success_chance + (current_skill / 20.0))
                if random.random() < success_chance:
                    self.creative_profile["skill_level"] = min(10.0, current_skill + random.uniform(0.1, 0.4))
                    self.creative_profile["experience_points"] += 10
                    reward += 15.0
                    self.available_opportunities["education"]["slots"] -= 1
                    self.creative_profile["current_engagement_status"] = 1
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 4.0
            else:
                reward -= 1.5

        elif action_name == "apply_gig_local":
            if self.available_opportunities["gigs"]["count"] > 0:
                base_success_chance = self.available_opportunities["gigs"]["quality"]
                success_chance = min(0.9, base_success_chance + (current_portfolio / 5.0) + (current_skill / 20.0))
                if random.random() < success_chance:
                    self.creative_profile["monetary_earnings_cumulative"] += random.uniform(100, 300)
                    self.creative_profile["social_media_reach"] = min(self.observation_space.spaces["social_media_reach"].high[0], current_reach + random.randint(20, 100))
                    self.creative_profile["experience_points"] += 5
                    reward += 10.0 # Low-tier gig reward
                    self.available_opportunities["gigs"]["count"] -= 1
                    self.creative_profile["current_engagement_status"] = 2 # Performing
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 7.0 # Penalty for rejection
            else:
                reward -= 3.0 # Penalty for no gigs

        elif action_name == "apply_gig_regional":
            if self.available_opportunities["gigs"]["count"] > 0:
                base_success_chance = self.available_opportunities["gigs"]["quality"]
                success_chance = min(0.8, base_success_chance + (current_portfolio / 3.0) + (current_skill / 10.0))
                if random.random() < success_chance:
                    self.creative_profile["monetary_earnings_cumulative"] += random.uniform(500, 1500)
                    self.creative_profile["social_media_reach"] = min(self.observation_space.spaces["social_media_reach"].high[0], current_reach + random.randint(100, 500))
                    self.creative_profile["experience_points"] += 20
                    reward += 30.0 # Mid-tier gig reward
                    self.available_opportunities["gigs"]["count"] -= 1
                    self.creative_profile["current_engagement_status"] = 2
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 15.0 # Penalty for rejection
            else:
                reward -= 5.0 # Penalty for no gigs

        elif action_name == "propose_brand_basic":
            if self.available_opportunities["brands"]["count"] > 0:
                base_success_chance = self.available_opportunities["brands"]["quality"]
                success_chance = min(0.7, base_success_chance + (current_reach / 10000.0) + (current_portfolio / 2.0))
                if random.random() < success_chance:
                    self.creative_profile["monetary_earnings_cumulative"] += random.uniform(2000, 5000)
                    self.creative_profile["social_media_reach"] = min(self.observation_space.spaces["social_media_reach"].high[0], current_reach + random.randint(500, 2000))
                    self.creative_profile["experience_points"] += 50
                    reward += 50.0 # Brand deal reward
                    self.available_opportunities["brands"]["count"] -= 1
                    self.creative_profile["current_engagement_status"] = 3 # Collaborating
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 25.0
            else:
                reward -= 10.0

        elif action_name == "schedule_investor_seed":
            if self.available_opportunities["investors"]["count"] > 0:
                base_success_chance = self.available_opportunities["investors"]["quality"]
                # Investor success depends heavily on earnings and portfolio quality
                success_chance = min(0.6, base_success_chance + (current_portfolio * 2) + (current_earnings / 50000.0))
                if random.random() < success_chance:
                    self.creative_profile["monetary_earnings_cumulative"] += random.uniform(10000, 50000)
                    self.creative_profile["social_media_reach"] = min(self.observation_space.spaces["social_media_reach"].high[0], current_reach + random.randint(1000, 5000))
                    self.creative_profile["experience_points"] += 100
                    reward += 100.0 # Investor deal reward
                    self.available_opportunities["investors"]["count"] -= 1
                    self.creative_profile["current_engagement_status"] = 3
                    self.creative_profile["time_since_last_engagement"] = 0
                else:
                    reward -= 50.0
            else:
                reward -= 20.0

        elif action_name == "focus_portfolio_improvement":
            self.creative_profile["portfolio_quality_score"] = min(1.0, current_portfolio + random.uniform(0.01, 0.03))
            self.creative_profile["experience_points"] += 5
            reward += 5.0 # Small positive for self-improvement
            self.creative_profile["current_engagement_status"] = 0 # Still idle but working
            self.creative_profile["time_since_last_engagement"] = 0

        elif action_name == "network_online":
            self.creative_profile["social_media_reach"] = min(self.observation_space.spaces["social_media_reach"].high[0], current_reach + random.randint(10, 50))
            self.creative_profile["experience_points"] += 2
            reward += 1.0 # Small positive for networking
            self.creative_profile["current_engagement_status"] = 0
            self.creative_profile["time_since_last_engagement"] = 0

        elif action_name == "rest_idle":
            # Small penalty to encourage action, but not too harsh
            reward -= 0.5
            self.creative_profile["current_engagement_status"] = 0 # Idle
            # No time_since_last_engagement reset here, as it's truly idle

        elif action_name == "explore_new_opportunities":
            self._update_opportunities() # Force an update of available opportunities
            reward += 2.0 # Small reward for actively seeking opportunities
            self.creative_profile["time_since_last_engagement"] = 0 # Considered an active step


        # --- Check Termination Conditions ---
        # Episode length limit
        if self.current_step >= self.max_steps:
            done = True
            # Add a bonus for reaching the end based on performance
            if self.creative_profile["monetary_earnings_cumulative"] > 50000:
                reward += 500 # Major success bonus
            elif self.creative_profile["monetary_earnings_cumulative"] < 5000:
                reward -= 200 # Significant failure penalty
            # print(f"Episode terminated: Max steps reached. Final Earnings: {self.creative_profile['monetary_earnings_cumulative']:.2f}")
        
        # Superstar status
        elif self.creative_profile["monetary_earnings_cumulative"] > 200000 and \
           self.creative_profile["social_media_reach"] > 50000 and \
           self.creative_profile["portfolio_quality_score"] > 0.95 and \
           self.creative_profile["skill_level"] > 9.0:
            reward += 1000 # Big bonus for reaching superstar status!
            done = True
            # print(f"Episode terminated: Superstar status achieved! Final Earnings: {self.creative_profile['monetary_earnings_cumulative']:.2f}")

        # Failure condition: Creative gives up due to prolonged inactivity and low performance
        elif self.creative_profile["time_since_last_engagement"] > 120 and \
             self.creative_profile["monetary_earnings_cumulative"] < 2000 and \
             self.creative_profile["skill_level"] < 3.0:
            reward -= 500 # Severe penalty for abandonment
            done = True
            # print(f"Episode terminated: Creative became inactive and gave up. Final Earnings: {self.creative_profile['monetary_earnings_cumulative']:.2f}")

        # Store last reward for rendering and accumulate total
        self.last_reward_value = reward
        self.total_episode_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.renderer.render(observation, self.last_action_idx, self.last_reward_value, self.total_episode_reward, self.current_step)
        
        # For rgb_array mode, the render() method will be called directly by SB3's EvalCallback
        # or when explicitly requested.
        
        return observation, reward, done, False, info # last False is for truncated

    def render(self):
        """
        This method is called by SB3's EvalCallback when render_mode="rgb_array".
        It returns the current frame as a numpy array.
        """
        if self.render_mode == "rgb_array" and self.renderer:
            return self.renderer.screen_to_rgb_array()
        else:
            # If not in rgb_array mode or renderer not initialized, return None
            return None 

    def close(self):
        if self.renderer:
            self.renderer.close()

# Example usage for testing the environment (can be put in main.py temporarily)
if __name__ == "__main__":
    env = TalentForgeConnectEnv(render_mode="human")
    obs, info = env.reset()
    print("Initial Observation:", {k: v.flatten() if isinstance(v, np.ndarray) else v for k, v in obs.items()})
    print("Initial Info:", info)

    terminated = False
    truncated = False
    total_reward = 0

    # Test with random actions
    for _ in range(500): # Run for 500 steps to see interaction
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print flattened observations for readability
        obs_flat = {k: v.flatten() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        # print(f"Step {env.current_step}: Action: {env.reverse_action_map[action]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Done: {terminated or truncated}")
        # print(f"  Skill: {obs_flat['skill_level'][0]:.2f}, Earnings: {obs_flat['monetary_earnings_cumulative'][0]:.2f}, Social Media: {obs_flat['social_media_reach'][0]:.0f}")

        if terminated or truncated:
            print(f"Episode finished at step {env.current_step}. Final Total Reward: {env.total_episode_reward:.2f}")
            break

    env.close()
