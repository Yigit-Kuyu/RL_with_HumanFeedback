# INFO

**RL_with_HumanFeedback_v1** defines a Partially Observable Markov Decision Process (POMDP) environment for a grid world and a Q-learning agent that incorporates human feedback to navigate the environment. The GridWorldPOMDP class simulates a 5x5 grid where an agent moves based on noisy observations, aiming to reach a goal at (4, 4). The QLearningAgent class is used to learn optimal actions by updating a Q-table based on the reward received from the environment and additional feedback from a simulated human. The agent explores the environment over multiple episodes, using an epsilon-greedy strategy to balance exploration and exploitation. 

**RLHF_withCovarienceMatrix_v2**  updates agent's belief based on actions taken and noisy observations, employing Kalman filter equations for belief updates. The step method allows the agent to take an action, receive a noisy observation, update its belief, and obtain a reward inversely proportional to the uncertainty (trace of the covariance matrix).

*RLHF_Main folder includes:*

- **RewardModelCreation** includes RewardModel class that is a neural network which processes states and actions through separate pathways, combines their representations, and predicts a reward for reinforcement learning. The reward_dataset class organizes training samples into tensors for states, actions, and rewards, and provides batches for training. The code also includes a training loop with early stopping to prevent overfitting.

- **RL_discrete** trains the RL agent, which involves storing past experiences in a replay memory, selecting actions based on an epsilon-greedy policy, and optimizing the Q-values through experience replay. Additionally, the training process incorporates a custom reward model that predicts rewards based on the state and action inputs. 



