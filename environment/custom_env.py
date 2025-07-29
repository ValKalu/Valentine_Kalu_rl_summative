# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.rendering import CreativeMindRenderer # Import your new renderer

class CreativeMindAcademyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Define the educational domains
    DOMAIN_NAMES = [
        "music_business",
        "legal_structures",
        "digital_marketing",
        "music_theory"
    ]

    def __init__(self, render_mode=None):
        super().__init__()

        # Define Action Space
        # Actions: Select a module, Attempt to answer a question, Study current module
        self.ACTION_MAP = {
            "select_module_music_business": 0,
            "select_module_legal_structures": 1,
            "select_module_digital_marketing": 2,
            "select_module_music_theory": 3,
            "attempt_answer": 4, # This action attempts to answer the current question
            "study_current_module": 5 # Agent can choose to study to increase knowledge
        }
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.reverse_action_map = {v: k for k, v in self.ACTION_MAP.items()} # For rendering

        # Define Observation Space (Using a dictionary for structured observation)
        self.observation_space = spaces.Dict({
            "current_module": spaces.Discrete(len(self.DOMAIN_NAMES)), # Index of the current module
            
            # Knowledge levels for each domain (0-100%)
            "knowledge_music_business": spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float32),
            "knowledge_legal_structures": spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float32),
            "knowledge_digital_marketing": spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float32),
            "knowledge_music_theory": spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float32),

            "questions_answered_correctly": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32),
            "questions_answered_incorrectly": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32),
            
            "current_question_difficulty": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32), # 0.0=Easy, 1.0=Hard
            "time_step": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32)
        })

        # Environment State Variables (Internal, will be reset at each episode)
        self.knowledge_levels = {} # Dictionary to store knowledge for each domain
        self.current_module_idx = 0 # Start with the first module
        self.correct_answers = 0
        self.incorrect_answers = 0
        self.current_question_difficulty = 0.5 # Default difficulty
        self.time_step = 0
        self.max_steps = 500 # Max steps per episode (e.g., 500 interactions)
        self.max_incorrect_streak = 5 # Max consecutive incorrect answers before termination
        self.incorrect_streak = 0

        # Rendering setup
        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = CreativeMindRenderer(self.DOMAIN_NAMES, self.ACTION_MAP, self.metadata["render_fps"])

        self.total_episode_score = 0.0 # Track total score for rendering

    def _get_obs(self):
        obs = {
            "current_module": self.current_module_idx,
            "questions_answered_correctly": np.array([self.correct_answers], dtype=np.int32),
            "questions_answered_incorrectly": np.array([self.incorrect_answers], dtype=np.int32),
            "current_question_difficulty": np.array([self.current_question_difficulty], dtype=np.float32),
            "time_step": np.array([self.time_step], dtype=np.int32)
        }
        for domain in self.DOMAIN_NAMES:
            obs[f"knowledge_{domain}"] = np.array([self.knowledge_levels[domain]], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "current_step": self.time_step,
            "total_score": self.total_episode_score,
            "correct_answers": self.correct_answers,
            "incorrect_answers": self.incorrect_answers
        }

    def _generate_question(self):
        """Generates a new question with difficulty based on current module knowledge."""
        current_domain = self.DOMAIN_NAMES[self.current_module_idx]
        current_knowledge = self.knowledge_levels[current_domain]

        # Difficulty tends to increase as knowledge in the module grows
        # But also introduce some randomness
        base_difficulty = current_knowledge / 100.0 # 0.0 (easy) to 1.0 (hard)
        self.current_question_difficulty = max(0.0, min(1.0, base_difficulty + random.uniform(-0.2, 0.2)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset knowledge levels to initial low state
        for domain in self.DOMAIN_NAMES:
            self.knowledge_levels[domain] = random.uniform(5.0, 15.0) # Start with some basic knowledge

        self.current_module_idx = random.randint(0, len(self.DOMAIN_NAMES) - 1) # Start in a random module
        self.correct_answers = 0
        self.incorrect_answers = 0
        self.time_step = 0
        self.total_episode_score = 0.0
        self.incorrect_streak = 0
        self._generate_question() # Generate the first question

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.renderer.render(observation, None, None, self.total_episode_score, self.time_step)

        return observation, info

    def step(self, action):
        reward = -0.1 # Small time penalty per step
        done = False
        self.time_step += 1

        action_name = self.reverse_action_map[action]
        self.last_action_idx = action # Store for rendering

        current_domain_name = self.DOMAIN_NAMES[self.current_module_idx]
        current_knowledge = self.knowledge_levels[current_domain_name]

        # --- Action Logic ---
        if action_name.startswith("select_module_"):
            new_module_name = action_name.replace("select_module_", "")
            new_module_idx = self.DOMAIN_NAMES.index(new_module_name)
            self.current_module_idx = new_module_idx
            self._generate_question() # Generate a new question for the selected module
            reward += 0.5 # Small reward for changing modules (exploration)
            self.incorrect_streak = 0 # Reset streak on module change

        elif action_name == "attempt_answer":
            # Probability of correct answer depends on knowledge vs. difficulty
            # Knowledge is 0-100, Difficulty is 0-1.0
            # A simple model: if knowledge is higher than (difficulty * 100), better chance
            success_chance = max(0.05, min(0.95, (current_knowledge - (self.current_question_difficulty * 100.0) + 50) / 100.0))
            
            if random.random() < success_chance:
                # Correct Answer!
                self.correct_answers += 1
                reward += 10.0 # High positive reward
                self.knowledge_levels[current_domain_name] = min(100.0, current_knowledge + random.uniform(1.0, 3.0) * (1.0 + self.current_question_difficulty)) # More knowledge gain for harder questions
                self.incorrect_streak = 0 # Reset streak
                # Generate a new question for the same module
                self._generate_question() 
            else:
                # Incorrect Answer
                self.incorrect_answers += 1
                reward -= 5.0 # Penalty for incorrect answer
                self.knowledge_levels[current_domain_name] = max(0.0, current_knowledge - random.uniform(0.5, 1.5)) # Small knowledge decay
                self.incorrect_streak += 1 # Increment streak
                # Generate a new question for the same module
                self._generate_question()

        elif action_name == "study_current_module":
            # Agent spends time studying, increasing knowledge but no immediate points
            self.knowledge_levels[current_domain_name] = min(100.0, current_knowledge + random.uniform(0.5, 2.0))
            reward += 2.0 # Small positive reward for studying
            self.incorrect_streak = 0 # Studying is a valid action, resets streak

        # --- Termination Conditions ---
        # 1. Max steps reached
        if self.time_step >= self.max_steps:
            done = True
            print(f"Episode terminated: Max steps reached. Final Score: {self.total_episode_score:.2f}")
            # Bonus/penalty based on overall performance
            if self.total_episode_score > 500:
                reward += 100
            elif self.total_episode_score < 0:
                reward -= 50

        # 2. All domains mastered (e.g., avg knowledge > 90%)
        avg_knowledge = sum(self.knowledge_levels.values()) / len(self.DOMAIN_NAMES)
        if avg_knowledge >= 90.0 and self.correct_answers >= 50: # Also need a minimum number of correct answers
            done = True
            reward += 500.0 # Big bonus for mastering
            print(f"Episode terminated: All domains mastered! Final Score: {self.total_episode_score:.2f}")

        # 3. Too many consecutive incorrect answers
        elif self.incorrect_streak >= self.max_incorrect_streak:
            done = True
            reward -= 100.0 # Significant penalty for giving up
            print(f"Episode terminated: Too many incorrect answers in a row. Final Score: {self.total_episode_score:.2f}")

        # Store last reward for rendering and accumulate total score
        self.last_reward_value = reward
        self.total_episode_score += reward

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.renderer.render(observation, self.last_action_idx, self.last_reward_value, self.total_episode_score, self.time_step)
        
        return observation, reward, done, False, info # last False is for truncated

    def render(self):
        """
        This method is called by SB3's EvalCallback when render_mode="rgb_array".
        It returns the current frame as a numpy array.
        """
        if self.render_mode == "rgb_array" and self.renderer:
            return self.renderer.screen_to_rgb_array()
        else:
            return None 

    def close(self):
        if self.renderer:
            self.renderer.close()

# Example usage for testing the environment
if __name__ == "__main__":
    # Add the shebang line if you intend to run this file directly like ./custom_env.py
    # #!/usr/bin/env python3

    # To run this, navigate to the project root directory (student_name_rl_summative)
    # and execute: python -m environment.custom_env
    
    env = CreativeMindAcademyEnv(render_mode="human")
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
        print(f"Step {env.time_step}: Action: {env.reverse_action_map[action]}, Reward: {reward:.2f}, Total Score: {env.total_episode_score:.2f}, Done: {terminated or truncated}")
        print(f"  Knowledge (MB): {obs_flat['knowledge_music_business'][0]:.1f}%, Correct: {obs_flat['questions_answered_correctly'][0]}, Incorrect: {obs_flat['questions_answered_incorrectly'][0]}")

        if terminated or truncated:
            print(f"Episode finished at step {env.time_step}. Final Total Score: {env.total_episode_score:.2f}")
            break

    env.close()

