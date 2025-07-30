# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import OrderedDict

# Import the rendering class
from environment.rendering import CreativeMindRenderer

class CreativeMindAcademyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.domain_names = ["music_business", "legal_structures", "digital_marketing", "music_theory"]
        self.num_domains = len(self.domain_names)
        self.max_knowledge = 100.0 # Max knowledge for any domain

        # Action Space:
        # 0-3: Select module (0=Music Business, 1=Legal Structures, etc.)
        # 4: Study current module
        # 5: Attempt answer
        self.ACTION_MAP = {
            "select_module_music_business": 0,
            "select_module_legal_structures": 1,
            "select_module_digital_marketing": 2,
            "select_module_music_theory": 3,
            "study_current_module": 4,
            "attempt_answer": 5
        }
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.reverse_action_map = {v: k for k, v in self.ACTION_MAP.items()}

        # Observation Space:
        # Dictionary of:
        # - 'knowledge_{domain}': Box(low=0, high=100, shape=(1,), dtype=np.float32) for each domain
        # - 'current_module': Discrete(num_domains)
        # - 'current_question_difficulty': Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # - 'questions_answered_correctly': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # - 'questions_answered_incorrectly': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        
        observation_space_dict = OrderedDict()
        for domain in self.domain_names:
            observation_space_dict[f'knowledge_{domain}'] = spaces.Box(low=0, high=self.max_knowledge, shape=(1,), dtype=np.float32)
        
        observation_space_dict['current_module'] = spaces.Discrete(self.num_domains)
        observation_space_dict['current_question_difficulty'] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        observation_space_dict['questions_answered_correctly'] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        observation_space_dict['questions_answered_incorrectly'] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(observation_space_dict)

        # Environment dynamics parameters
        self.study_gain_per_step = 0.5 # Knowledge gain per study action
        self.max_episode_steps = 750 # Maximum steps per episode
        self.mastery_bonus_threshold = 95.0 # Knowledge % for mastery bonus
        self.mastery_bonus_reward = 100.0 # Reward for achieving mastery in a domain
        self.question_attempt_cost = -0.5 # Small penalty for attempting a question

        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None # Will be initialized in _render_frame if render_mode is set

    def _get_obs(self):
        obs = OrderedDict()
        for domain in self.domain_names:
            obs[f'knowledge_{domain}'] = np.array([self.knowledge[domain]], dtype=np.float32)
        
        obs['current_module'] = self.current_module_idx
        obs['current_question_difficulty'] = np.array([self.current_question_difficulty], dtype=np.float32)
        obs['questions_answered_correctly'] = np.array([self.questions_answered_correctly], dtype=np.float32)
        obs['questions_answered_incorrectly'] = np.array([self.questions_answered_incorrectly], dtype=np.float32)
        return obs

    def _get_info(self):
        # Additional info for debugging or analysis, if needed
        return {
            "total_score": self.total_score,
            "episode_step": self.current_step,
            "current_module_name": self.domain_names[self.current_module_idx]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize knowledge for all domains randomly between 0 and 50
        self.knowledge = {domain: random.uniform(0, 50) for domain in self.domain_names}
        
        # Randomly select initial module
        self.current_module_idx = self.np_random.integers(0, self.num_domains)
        
        # Randomly assign initial question difficulty (0.0=easy, 1.0=hard)
        self.current_question_difficulty = self.np_random.uniform(0.0, 1.0)

        self.questions_answered_correctly = 0
        self.questions_answered_incorrectly = 0
        self.total_score = 0.0
        self.current_step = 0
        self.mastered_domains = {domain: False for domain in self.domain_names} # Track mastery

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        self.current_step += 1

        action_name = self.reverse_action_map[action]
        current_domain = self.domain_names[self.current_module_idx]

        if "select_module" in action_name:
            # Action: Select a new module
            selected_domain_idx = self.ACTION_MAP[action_name] # Get the domain index from the action map
            self.current_module_idx = selected_domain_idx
            reward += 0.4 # Small reward for exploring/changing modules
        elif action_name == "study_current_module":
            # Action: Study current module
            # Gain knowledge, capped at max_knowledge
            self.knowledge[current_domain] = min(self.max_knowledge, self.knowledge[current_domain] + self.study_gain_per_step)
            reward += 2.0 + (self.knowledge[current_domain] / self.max_knowledge) * 0.5 # Scaled reward for studying
            
            # Check for mastery bonus
            if not self.mastered_domains[current_domain] and self.knowledge[current_domain] >= self.mastery_bonus_threshold:
                reward += self.mastery_bonus_reward
                self.mastered_domains[current_domain] = True
                print(f"MASTERED DOMAIN: {current_domain}! Bonus: {self.mastery_bonus_reward}") # Debug print
        elif action_name == "attempt_answer":
            # Action: Attempt to answer a question
            reward += self.question_attempt_cost # Apply cost for attempting

            # Calculate success probability based on current module knowledge and question difficulty
            # Higher knowledge means higher chance of success, higher difficulty means lower chance
            knowledge_effect = self.knowledge[current_domain] / self.max_knowledge
            difficulty_effect = 1.0 - self.current_question_difficulty # Invert difficulty for easier calculation
            
            # Simple probability model: knowledge helps, difficulty hurts
            success_probability = (knowledge_effect + difficulty_effect) / 2.0
            success_probability = np.clip(success_probability, 0.1, 0.9) # Clamp between 10% and 90%

            if random.random() < success_probability:
                # Correct answer
                correct_reward = 10.0 + (1.0 - self.current_question_difficulty) * 5.0 # Easier questions give less, harder more
                reward += correct_reward
                self.questions_answered_correctly += 1
                # Increase difficulty for next question in this module after correct answer
                self.current_question_difficulty = min(1.0, self.current_question_difficulty + 0.1)
            else:
                # Incorrect answer
                incorrect_penalty = -5.0 - (self.current_question_difficulty * 5.0) # Harder questions penalize more
                reward += incorrect_penalty
                self.questions_answered_incorrectly += 1
                # Decrease difficulty for next question in this module after incorrect answer
                self.current_question_difficulty = max(0.0, self.current_question_difficulty - 0.1)
            
            # After attempting, always generate a new question difficulty for the current module
            # This ensures variety and prevents getting stuck on one difficulty.
            self.current_question_difficulty = self.np_random.uniform(0.0, 1.0)


        self.total_score += reward

        # Termination conditions
        # If all domains are mastered
        if all(self.mastered_domains.values()):
            terminated = True
            reward += 500.0 # Large bonus for completing all domains
            print("All domains mastered! Episode terminated.") # Debug print

        # Episode timeout
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print("Episode truncated due to max steps.") # Debug print

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # Pass relevant data to the renderer
            self.render(last_action=action, last_reward=reward, total_score=self.total_score, episode_step=self.current_step)

        return observation, reward, terminated, truncated, info

    # --- UPDATED RENDER METHOD SIGNATURE ---
    def render(self, last_action=None, last_reward=None, total_score=None, episode_step=None):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without setting render_mode "
                "in your environment's initialization. "
                "Pass render_mode='human' or render_mode='rgb_array' to the CreativeMindAcademyEnv constructor."
            )
            return

        if self.renderer is None:
            self.renderer = CreativeMindRenderer(self.domain_names, self.ACTION_MAP, self.metadata["render_fps"])

        # Pass current observation and other dynamic data to the renderer
        # Ensure observation is generated fresh for rendering
        current_obs = self._get_obs() 
        self.renderer.render(
            observation=current_obs,
            last_action_idx=last_action, # Pass the scalar action directly
            reward_value=last_reward,
            total_score=total_score if total_score is not None else self.total_score,
            episode_step=episode_step if episode_step is not None else self.current_step
        )
        
        if self.render_mode == "rgb_array":
            return self.renderer.screen_to_rgb_array()
        
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

