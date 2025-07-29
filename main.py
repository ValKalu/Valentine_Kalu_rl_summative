# main.py
#!/usr/bin/env python3 # Shebang for direct execution

import gymnasium as gym
import os
import datetime
import imageio # For GIF creation and video recording
import numpy as np
import matplotlib.pyplot as plt

# SB3 imports
from stable_baselines3 import DQN, PPO, A2C # Import other models as you implement their training
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Import your custom environment
from environment.custom_env import CreativeMindAcademyEnv
from environment.rendering import get_action_name # For cleaner action display in GIFs/videos

# --- Configuration ---
MODELS_DIR_BASE = "models"
LOGS_DIR_BASE = "logs"
VIDEO_OUTPUT_DIR = "videos"
GIF_OUTPUT_DIR = "gifs"
STATIC_VIS_OUTPUT_DIR = "static_visualizations"

# Ensure all necessary directories exist
os.makedirs(MODELS_DIR_BASE, exist_ok=True)
os.makedirs(LOGS_DIR_BASE, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_VIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR_BASE, "dqn"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR_BASE, "ppo"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR_BASE, "a2c"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR_BASE, "reinforce"), exist_ok=True) # Placeholder for REINFORCE

def generate_random_action_visualization(env_class, num_steps=300, output_path=None, fps=30):
    """
    Generates a GIF of the environment with random actions.
    This demonstrates the visualization without any trained model.
    """
    print(f"Generating random action visualization for {num_steps} steps...")
    env = env_class(render_mode="rgb_array") # Use rgb_array for capturing frames
    obs, info = env.reset()
    frames = []
    
    # Get action map from environment instance
    action_map = env.ACTION_MAP 

    for step in range(num_steps):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture the frame after the step and rendering logic within env.step
        frame = env.render() 
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            print(f"Random action visualization stopped at step {step} due to episode termination.")
            break
    
    env.close()

    if output_path and frames:
        imageio.mimsave(output_path, frames, fps=fps) # Adjust fps as needed
        print(f"Random action GIF saved to {output_path}")
    elif not frames:
        print("No frames captured for random action GIF. Check environment rendering.")
    else:
        print("No output path provided for random action GIF.")

def train_agent(model_type, env_class, total_timesteps=1_000_000, reward_threshold=500.0):
    """
    Trains an RL agent of the specified type.
    """
    model_name = model_type.lower()
    log_dir = os.path.join(LOGS_DIR_BASE, model_name)
    models_dir = os.path.join(MODELS_DIR_BASE, model_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n--- Starting {model_type} training ---")

    # Create environment wrapped with Monitor for logging
    # Monitor wrapper saves results to a CSV file in log_dir
    env = Monitor(env_class(), log_dir) 
    vec_env = make_vec_env(lambda: env_class(), n_envs=1, seed=0) # For SB3 compatibility

    # Callbacks
    # Stop training if the mean reward over evaluation episodes reaches a threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    
    # Evaluate the agent periodically and save the best model
    eval_env = Monitor(env_class(), log_dir) # Separate eval env, also monitored
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 best_model_save_path=models_dir,
                                 log_path=log_dir, # Logs evaluation results here
                                 eval_freq=1000, # Evaluate every 1000 steps
                                 n_eval_episodes=5, # Number of episodes for evaluation
                                 deterministic=True, # Use deterministic actions during evaluation
                                 render=False) # Do not render during evaluation to speed it up

    # Model Initialization and Hyperparameters (TWEAK THESE!)
    # These are starting points; you MUST experiment with them.
    if model_type == "DQN":
        model = DQN("MultiInputPolicy", # Use MultiInputPolicy for Dict observation space
                    vec_env,
                    learning_rate=1e-4, # Common value, try 1e-3, 5e-5
                    buffer_size=50000, # Large buffer for stability, try 10000, 100000
                    learning_starts=1000, # Number of steps to populate buffer before learning, try 0, 5000
                    batch_size=32, # Batch size for gradient updates, try 64, 128
                    gamma=0.99, # Discount factor, try 0.9, 0.95
                    exploration_fraction=0.1, # Fraction of total timesteps for annealing epsilon
                    exploration_final_eps=0.05, # Final value of epsilon during exploration
                    target_update_interval=1000, # Update target network every `target_update_interval` steps
                    verbose=1, # Print training information
                    tensorboard_log=LOGS_DIR_BASE) # Log to base logs dir, TensorBoard will pick up subdirs
    elif model_type == "PPO":
        model = PPO("MultiInputPolicy",
                    vec_env,
                    learning_rate=3e-4, # Common value, try 1e-3, 5e-5
                    n_steps=2048, # Number of steps to run for each environment per update
                    batch_size=64, # Minibatch size for SGD
                    n_epochs=10, # Number of epochs when optimizing the surrogate loss
                    gamma=0.99, # Discount factor
                    gae_lambda=0.95, # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                    clip_range=0.2, # Clipping parameter for PPO's policy update
                    ent_coef=0.01, # Entropy coefficient for exploration
                    verbose=1,
                    tensorboard_log=LOGS_DIR_BASE)
    elif model_type == "A2C":
        model = A2C("MultiInputPolicy",
                    vec_env,
                    learning_rate=7e-4, # Common value, try 1e-3, 5e-4
                    n_steps=5, # Number of steps to run for each environment per update
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    verbose=1,
                    tensorboard_log=LOGS_DIR_BASE)
    elif model_type == "REINFORCE":
        # SB3 does not have a direct REINFORCE implementation.
        # A common approximation is A2C with n_steps=1 (equivalent to Monte Carlo policy gradient)
        # However, for a pure REINFORCE comparison, you might need a custom training loop or another library.
        # For this assignment, if you must use SB3, A2C with n_steps=1 is the closest.
        # Be sure to explain this implementation detail in your report.
        print("Note: SB3 approximates REINFORCE with A2C(n_steps=1). Be aware of its limitations.")
        model = A2C("MultiInputPolicy",
                    vec_env,
                    learning_rate=5e-4,
                    n_steps=1, # Key for REINFORCE-like behavior (Monte Carlo policy gradient)
                    gamma=0.99,
                    gae_lambda=1.0, # For pure Monte Carlo, gae_lambda=1.0 (no TD-error estimation)
                    ent_coef=0.0, # REINFORCE typically doesn't have entropy bonus (can add if desired for exploration)
                    verbose=1,
                    tensorboard_log=LOGS_DIR_BASE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Training
    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=eval_callback,
                    log_interval=10) # Log training progress every 10 updates
        print(f"{model_type} training finished.")
    except StopTrainingOnRewardThreshold:
        print(f"Training for {model_type} stopped due to reward threshold being met.")
    except KeyboardInterrupt:
        print(f"Training for {model_type} interrupted by user.")

    # Save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    final_model_path = os.path.join(models_dir, f"{model_name}_final_model_{timestamp}")
    model.save(final_model_path)
    print(f"Final {model_type} model saved to {final_model_path}.zip")

    # Close environments
    env.close()
    eval_env.close()
    vec_env.close()


def evaluate_and_record_agent(model_path, env_class, n_episodes=3, fps=30, output_filename="evaluation_video.mp4"):
    """
    Evaluates a trained agent and records its performance over multiple episodes.
    """
    print(f"\n--- Evaluating agent from {model_path} and recording {n_episodes} episodes ---")
    
    # Determine model type from path to load correctly
    model_type_str = os.path.basename(os.path.dirname(model_path))
    
    if "dqn" in model_type_str:
        ModelClass = DQN
    elif "ppo" in model_type_str:
        ModelClass = PPO
    elif "a2c" in model_type_str or "reinforce" in model_type_str: # A2C is used for REINFORCE
        ModelClass = A2C
    else:
        print(f"Could not determine model type from path: {model_path}. Skipping evaluation.")
        return

    model = ModelClass.load(model_path)
    env = env_class(render_mode="rgb_array") # Ensure rgb_array mode for video recording

    video_writer = imageio.get_writer(os.path.join(VIDEO_OUTPUT_DIR, output_filename), fps=fps)

    action_map = env.ACTION_MAP # Get action map from env for display

    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        print(f"--- Starting Episode {i+1} ---")

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True) # Use deterministic policy for evaluation
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            frame = env.render() # Your env.render() should return the numpy array frame
            if frame is not None:
                video_writer.append_data(frame)

            if done or truncated:
                print(f"Episode {i+1} finished. Total Reward: {episode_reward:.2f}, Steps: {step_count}")

    video_writer.close()
    env.close()
    print(f"Evaluation video saved to {os.path.join(VIDEO_OUTPUT_DIR, output_filename)}")

def plot_results(log_folder, title='Learning Curve'):
    """
    Plots the results from the monitor log file.
    """
    print(f"\n--- Generating plot for {log_folder} ---")
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    if len(x) == 0:
        print(f"No data to plot for {log_folder}. Ensure training ran and logs were created.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Rewards')
    ax.set_title(title)
    ax.grid(True)
    
    plot_filename = os.path.join(log_folder, f"{os.path.basename(log_folder)}_learning_curve.png")
    plt.savefig(plot_filename)
    print(f"Learning curve plot saved to {plot_filename}")
    plt.close(fig) # Close the plot to free memory


if __name__ == "__main__":
    print("Welcome to CreativeMind Academy RL Project!")

    # --- Task 1: Generate Static Visualization (GIF of random actions) ---
    # This GIF demonstrates your environment's visualization without any trained model.
    random_gif_path = os.path.join(GIF_OUTPUT_DIR, "creativemind_random_actions.gif")
    generate_random_action_visualization(CreativeMindAcademyEnv, num_steps=300, output_path=random_gif_path)
    print(f"\n**Action Required:** Remember to add '{random_gif_path}' to your report.")

    # --- Task 2: Train Agents ---
    # Uncomment and run these one by one.
    # Adjust `total_timesteps` and `reward_threshold` based on your environment's complexity
    # and desired performance.
    # To monitor training: open a new terminal in your project root and run `tensorboard --logdir logs`

    # Example: Train DQN Agent
    # print("\n--- Training DQN Agent ---")
    # train_agent("DQN", CreativeMindAcademyEnv, total_timesteps=200_000, reward_threshold=1000.0)
    # plot_results(os.path.join(LOGS_DIR_BASE, "dqn"), "DQN Learning Curve")

    # Example: Train PPO Agent (Uncomment to run)
    # print("\n--- Training PPO Agent ---")
    # train_agent("PPO", CreativeMindAcademyEnv, total_timesteps=200_000, reward_threshold=1500.0)
    # plot_results(os.path.join(LOGS_DIR_BASE, "ppo"), "PPO Learning Curve")

    # Example: Train A2C Agent (Uncomment to run)
    # print("\n--- Training A2C Agent ---")
    # train_agent("A2C", CreativeMindAcademyEnv, total_timesteps=200_000, reward_threshold=800.0)
    # plot_results(os.path.join(LOGS_DIR_BASE, "a2c"), "A2C Learning Curve")

    # Example: Train REINFORCE-like Agent (using A2C with n_steps=1) (Uncomment to run)
    # print("\n--- Training REINFORCE-like Agent ---")
    # train_agent("REINFORCE", CreativeMindAcademyEnv, total_timesteps=200_000, reward_threshold=700.0)
    # plot_results(os.path.join(LOGS_DIR_BASE, "reinforce"), "REINFORCE Learning Curve")


    # --- Task 3: Evaluate and Record Trained Agents ---
    # AFTER you have trained models and they are saved in models/dqn, models/ppo, etc.
    # The EvalCallback usually saves a 'best_model.zip' in the models/ALGORITHM_NAME directory.
    # You might need to manually verify the path if you save models with different names.

    # Example: Evaluate and record DQN agent
    dqn_models_path = os.path.join(MODELS_DIR_BASE, "dqn")
    best_dqn_model_path = os.path.join(dqn_models_path, "best_model.zip") # Default name from EvalCallback
    
    if os.path.exists(best_dqn_model_path):
       evaluate_and_record_agent(best_dqn_model_path, CreativeMindAcademyEnv, n_episodes=3, output_filename="dqn_agent_performance.mp4")
    else:
       print(f"\nNo DQN best model found at {best_dqn_model_path}. Please train it first by uncommenting the train_agent('DQN', ...) line.")

    # Example: Evaluate and record PPO agent (Uncomment to run)
    # ppo_models_path = os.path.join(MODELS_DIR_BASE, "ppo")
    # best_ppo_model_path = os.path.join(ppo_models_path, "best_model.zip")
    # if os.path.exists(best_ppo_model_path):
    #     evaluate_and_record_agent(best_ppo_model_path, CreativeMindAcademyEnv, n_episodes=3, output_filename="ppo_agent_performance.mp4")
    # else:
    #     print(f"\nNo PPO best model found at {ppo_models_path}. Please train it first.")

    # Repeat for A2C and REINFORCE (using A2C)
    # a2c_models_path = os.path.join(MODELS_DIR_BASE, "a2c")
    # best_a2c_model_path = os.path.join(a2c_models_path, "best_model.zip")
    # if os.path.exists(best_a2c_model_path):
    #     evaluate_and_record_agent(best_a2c_model_path, CreativeMindAcademyEnv, n_episodes=3, output_filename="a2c_agent_performance.mp4")
    # else:
    #     print(f"\nNo A2C best model found at {a2c_models_path}. Please train it first.")

    # reinforce_models_path = os.path.join(MODELS_DIR_BASE, "reinforce")
    # best_reinforce_model_path = os.path.join(reinforce_models_path, "best_model.zip")
    # if os.path.exists(best_reinforce_model_path):
    #     evaluate_and_record_agent(best_reinforce_model_path, CreativeMindAcademyEnv, n_episodes=3, output_filename="reinforce_agent_performance.mp4")
    # else:
    #     print(f"\nNo REINFORCE best model found at {reinforce_models_path}. Please train it first.")

