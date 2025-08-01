# Valentine_Kalu_rl_summative


# Reinforcement Learning for Creative Professionals in the African Music Industry

This project is a summative assignment for the Machine Learning Techniques II course at the African Leadership University. It explores the application of Reinforcement Learning (RL) to create a personalized, adaptive educational platform for African music creatives. The goal is to provide tailored guidance on the business and technical aspects of the music industry, such as distribution, copyright, and marketing.

# 1. Project Overview
The African music industry is a rapidly growing sector, but many creatives lack access to structured education on the business side of their craft. This project addresses this gap by developing a reinforcement learning agent that acts as a personalized educational assistant. The agent's core purpose is to recommend learning modules, resources, and next steps to a user in a way that maximizes their long-term learning and engagement.

The agent learns from user interactions, adapting its suggestions based on the user's progress, strengths, and weaknesses to create a dynamic and effective learning path.

# 2. Environment Description
2.1 Agent(s)
The agent in this environment is the Reinforcement Learning model itself. It's a "pedagogical agent" whose role is to observe the user's state and take actions that guide the user's learning journey.

2.2 Action Space
The action space is a set of discrete, high-level educational interventions that the agent can perform. These actions are designed to influence the user's learning path positively. Examples include:

Suggest Module: Music Distribution

Suggest Module: Copyright & Licensing

Suggest Module: Social Media Marketing

Provide a Resource Link

Ask a Clarifying Question

Recommend a Quiz

2.3 State Space
The state space represents the user's current educational context. The agent's observations are encoded into a vector that includes:

User Profile: (e.g., music genre, experience level)

Progress Metrics: (e.g., percentage of modules completed, quiz scores)

Engagement Data: (e.g., time spent on a module, number of skips)

Knowledge Representation: (e.g., a vector representing the user's current knowledge in various topics)

Recent Interaction History: (e.g., the last action taken by the agent, the last user response)

2.4 Reward Structure
The reward function is designed to incentivize the agent to make decisions that lead to successful learning outcomes for the user. Rewards are positive for behaviors that indicate engagement and knowledge gain, and negative for those that suggest disinterest or frustration.

+1 for successful completion of a module.

+0.5 for a correct answer on a quiz question.

+5 for a user providing positive feedback on an agent's suggestion.

-1 for a user skipping a recommended module.

-5 for a user providing negative feedback or disengaging from the platform.

A small negative reward (-0.1) for each time step to encourage efficient learning.

2.5 Environment Visualization
The environment visualization is the user interface (UI) of the educational platform. It consists of:

A main dashboard showing the user's progress and recommended learning path.

An interactive chat interface where the agent provides personalized advice.

A library of learning modules (e.g., text, video, quizzes).

A progress tracker displaying completed sections and knowledge gaps.

3. Implemented Methods
This project will implement and evaluate the following reinforcement learning algorithms to determine the most effective approach for the pedagogical agent:

Deep Q-Network (DQN): An algorithm that learns a policy for selecting the optimal next action (e.g., which module to suggest) by approximating the action-value function using a deep neural network. The use of a replay buffer and a target network will be key to ensuring stable training.

REINFORCE: A policy-gradient method that directly optimizes the policy network based on the cumulative reward received. This method will explore how directly rewarding good outcomes can shape the agent's behavior.

Proximal Policy Optimization (PPO): A more advanced policy-gradient method known for its stability. PPO will be used to see if we can achieve better and more consistent performance compared to REINFORCE by preventing large policy updates.

Advantage Actor-Critic (A2C): An algorithm that combines value-based and policy-based methods. The actor suggests actions, and the critic evaluates the state to provide feedback, potentially leading to faster and more stable learning.

📊 Hyperparameter Tuning Summary
DQN Agent Hyperparameters

| Hyperparameter        | Optimal Value | Summary                                                                 |
|-----------------------|---------------|-------------------------------------------------------------------------|
| Learning Rate         | 0.00025       | Controls how much the model weights are updated during training         |
| Gamma (γ)             | 0.99          | Discount factor that prioritizes future rewards                         |
| Replay Buffer Size    | 1,000,000     | Stores past experiences for training stability                          |
| Batch Size            | 32            | Number of samples used per training step                                |
| Target Network Update | Every 10,000  | Frequency at which the target network is updated                        |
| Exploration Strategy  | ε-greedy      | Agent initially explores (ε=1.0) and decays ε over time to exploit more |
| ε Decay               | 0.995         | Rate at which ε is reduced after each step                              |
| Minimum ε             | 0.01          | The lowest exploration rate allowed                                     |
| Max Steps             | 1,000,000     | Total number of training steps                                          |

# REINFORCE Agent Hyperparameters

| Hyperparameter  | Optimal Value | Summary                                          |
|-----------------|---------------|--------------------------------------------------|
| Learning Rate   | 0.001         | Step size for updating policy parameters         |
| Gamma (γ)       | 0.99          | Future reward discount factor                    |
| Batch Size      | 5 episodes    | Number of episodes per update                    |
| Entropy Bonus   | 0.01          | Encourages exploration by discouraging certainty |

### PPO Agent Hyperparameters

| Hyperparameter        | Optimal Value | Summary                                                       |
|------------------------|---------------|---------------------------------------------------------------|
| Learning Rate          | 0.0003        | Step size for policy update                                   |
| Clip Range             | 0.2           | Limits the change in policy update to improve stability       |
| Gamma (γ)              | 0.99          | Discount rate for future rewards                              |
| GAE Lambda             | 0.95          | Trade-off between bias and variance in advantage estimation   |
| Update Epochs          | 4             | Number of policy updates per batch                            |
| Minibatch Size         | 64            | Mini-batch size within each epoch                             |
| Entropy Coefficient    | 0.01          | Encourages policy randomness                                  |

### A2C Agent Hyperparameters

| Hyperparameter     | Optimal Value | Summary                                               |
|--------------------|---------------|--------------------------------------------------------|
| Learning Rate      | 0.0007        | Learning rate for updating actor-critic networks       |
| Gamma (γ)          | 0.99          | Reward discount factor                                 |
| Value Loss Coeff.  | 0.5           | Weight of value function loss                          |
| Entropy Coefficient| 0.01          | Encourages exploration                                 |
| n-step Return      | 5             | Number of steps to look ahead for bootstrapping        |

# Clone repository:

git clone  (repo link )
cd cartpole-drl

# Create a virtual environment:

python -m venv drl_env
source drl_env/bin/activate  # On Linux/macOS

# Install dependencies:
pip install -r requirements.txt

# 🎥 Agent Demonstration
A video showing the agent maximizing rewards over 3 episodes is included here: [Watch Agent Simulation CartPole-v1](https://www.loom.com/share/d532136ef9b746af95ba0fd075a25bcb?sid=c22e3564-9306-477f-8bf0-14d3917764f8)

📚 References
OpenAI Gym

Stable-Baselines3

Sutton & Barto, Reinforcement Learning: An Introduction

Spinning Up by OpenAI

👤 Author
Valentine Kalu
Machine Learning Engineer 