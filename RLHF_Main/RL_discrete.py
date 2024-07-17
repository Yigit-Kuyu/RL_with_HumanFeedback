import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.to(self.device)

    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def select_action(self, state):
        eps_threshold = self.epsilon_threshold()
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # Select the action with the highest Q-value
                return self(state).max(1)[1].view(1, 1)
        else:
            # Select a random action
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        
    def epsilon_threshold(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-1. * self.steps_done / self.eps_decay)


# Call the saved reward model.
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=100, output_dim=1) -> None:
        super().__init__()
        self.state_dim = state_dim # dim=4
        self.action_dim = action_dim #dim=1
        
        self.state_processor = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(True),
        )
        self.action_processor = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 8),  
            nn.Dropout(0.4),
            nn.ReLU(True),
            nn.Linear(8, output_dim),
        )

    def forward(self, sx, ax):
        batch_size=sx.size(0)
        # Reshape s and a to have the correct dimensions
        st = sx.view(batch_size, -1)  # st.shape [4,4]
        ac = ax.view(batch_size, -1)  # ac.shape [4,1]
        

        s = self.state_processor(st)  # s.shape [4,100]
        a = self.action_processor(ac) # a.shape [4,100]
        combined = torch.cat([s, a], dim=1)
        output=self.output_layer(combined).reshape(-1, 1)
        return output

# Function to plot durations and cumulative rewards
def plot_metrics(title, durations, rewards):
    #plt.figure(figsize=(12, 8))
    plt.figure(1)
    plt.clf()

    # Subplot for episode durations
    plt.subplot(2, 1, 1)
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration (number of time steps taken)')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Subplot for cumulative rewards
    plt.subplot(2, 1, 2)
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)



def train_dqn_step():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Unpack the transitions
    state_batch, action_batch, next_state_batch, reward_batch = zip(*transitions)

    # Create a mask to filter out non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=device, dtype=torch.bool)
    # Concatenate only the non-final next states
    non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])

    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(reward_batch)

    current_q_values = policy_net(state_batch) # produce Q-values for all actions for each state in the batch. torch.Size([128, 2])
    sa_q_values = current_q_values.gather(1, action_batch) # Q-values for the actions taken torch.Size([128, 1])

   
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # contains the maximum Q-value for the next state for each state in the batch. torch.Size([128])
   
    expected_sa_q_values = (next_state_values * gamma) + reward_batch.squeeze()  # Compute the expected Q values torch.Size([128])
    expected_state_action_q_values = expected_sa_q_values.unsqueeze(1)  # Ensure expected_state_action_values is of shape [batch_size, 1]

    
    criterion = nn.SmoothL1Loss() # Compute Huber loss
    loss = criterion(sa_q_values, expected_state_action_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


batch_size = 128 # the number of transitions sampled from the replay buffer
gamma= 0.99 # the discount factor
tau= 0.005 # the update rate of the target network
lr = 1e-4 # the learning rate
rp_m=1000 # replay memory size 

env = gym.make("CartPole-v1", render_mode="human")
# Get number of actions from gym action space
state,info = env.reset()
n_observations =env.observation_space.shape[0]
n_actions = env.action_space.n # # This should be 2
n_human_actions=1
print("state dimension: ", n_observations) # 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

RM = RewardModel(n_observations, n_human_actions).cuda()
RM.load_state_dict(torch.load("reward_model.pkl"))

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
# RM_optimizer = optim.Adam(RM.parameters(), lr=0.001)

memory = ReplayMemory(rp_m)

steps_done = 0 
episode_durations = []
episode_rewards = []  
num_train_episodes = 10
num_test_episodes=20

######################################## Training #######################################

for i_episode in range(num_train_episodes):
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulative_reward = 0  # Initialize cumulative reward for the episode
    t = 0  # Initialize time step counter
    
    while True:
        t += 1  # Increment time step
        action = policy_net.select_action(state)
        observation, _, terminated, truncated, _ = env.step(action.item())
        with torch.no_grad():
            reward = RM(state.to(device).reshape(1, 4), action.to(device, dtype=torch.float32).reshape(1, 1))
            reward = reward.to(device) 
    
        done = terminated or truncated
        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward) # Store the transition in memory
        state = next_state
        cumulative_reward += reward.item()  # Accumulate reward
        train_dqn_step() # Optimization on the policy network
        target_net_state_dict = target_net.state_dict() # Soft update of the target network's weights
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t)
            episode_rewards.append(cumulative_reward)  # Append cumulative reward
            plot_metrics('Training Process',episode_durations, episode_rewards)
            break

        

print('Training Complete')
plt.close()  # Close the training figure
plt.ioff()
plt.show()



torch.save({
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'episode_durations': episode_durations,
    'episode_rewards': episode_rewards
}, 'RL_Model.pkl.tar')


######################################## Testing #######################################

# Reset lists for testing data
test_episode_durations = []
test_episode_rewards = []

policy_net = DQN(n_observations, n_actions).cuda()
print('Test data looding....')
checkpoint = torch.load('RL_Model.pkl.tar')
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
print('Test data looded')
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for episode in range(num_test_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulative_reward = 0
    t = 0

    while True:
        env.render()
        with torch.no_grad():
            action = policy_net(state.cuda()).max(1)[1].view(1, 1)
        print('action taken: ', action.item())
        observation, reward, terminated, truncated, info = env.step(action.item())
        cumulative_reward += reward
        t += 1
        if done:
            print('Episode ended')
            test_episode_durations.append(t)
            test_episode_rewards.append(cumulative_reward)
            plot_metrics('Testing Process',test_episode_durations, test_episode_rewards)
            break
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env.close()
print('Testing Complete')
