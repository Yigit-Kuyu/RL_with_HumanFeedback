import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



# creating the reward model.
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=100, output_dim=1) -> None:
        super().__init__()
        self.state_dim = state_dim # dim=4
        self.action_dim = action_dim #dim=1
        
        self.state_processor = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Dropout(0.23),
            nn.ReLU(True),
        )
        self.action_processor = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 8),  # Combining state and action hidden dimensions
            nn.Dropout(0.44),
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
    


class reward_dataset:
    def __init__(self, samples):
        self.samples = samples
        self.preprocess_data()

    def preprocess_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        for sample in self.samples:
            state, action = sample[0]
            reward = sample[1]
            self.states.append(torch.tensor(state, dtype=torch.float32).reshape(1, 4))
            self.actions.append(torch.tensor(action, dtype=torch.float32).reshape(1, 1))
            self.rewards.append(torch.tensor(reward, dtype=torch.float32).reshape(1, 1))

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        batch_states = torch.cat([self.states[i] for i in indices])
        batch_actions = torch.cat([self.actions[i] for i in indices])
        batch_rewards = torch.cat([self.rewards[i] for i in indices])
        return batch_states, batch_actions, batch_rewards


def get_human_feedback():
    feedback = random.randint(0, 9)
    return [feedback]


# Training and validation functions
def train_epoch(model, data, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(data.samples) // batch_size
    for _ in range(num_batches):
        s, a, r = data.get_batch(batch_size)
        s, a, r = s.cuda(), a.cuda(), r.cuda()
        
        pred = model(s, a)
        loss = F.mse_loss(pred, r)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / num_batches

def validate(model, data):
    model.eval()
    total_loss = 0
    num_batches = len(data.samples) // batch_size
    with torch.no_grad():
        for _ in range(num_batches):
            s, a, r = data.get_batch(batch_size)
            s, a, r = s.cuda(), a.cuda(), r.cuda()
            pred = model(s, a)
            loss = F.mse_loss(pred, r)
            total_loss += loss.item()
    return total_loss / num_batches



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the environment.
#env = gym.make("CartPole-v1", render_mode="rgb_array")
#env = gym.make("CartPole-v1")
env = gym.make("CartPole-v1", render_mode="human")



state,info = env.reset()
n_observations =env.observation_space.shape[0]
# n_actions = env.action_space.n # # This should be 2, but it's not used in the rewardz model
n_human_actions=1 # since human gives one rank for the corresponding state, it should be 1.
print("state dimension: ", n_observations) # 4

# Initialize model and optimizer
RM = RewardModel(n_observations, n_human_actions).cuda()
RM_optimizer = optim.SGD(RM.parameters(), lr=1e-3)

# Training parameters
EPOCHS = 15
batch_size = 4


# Model should generalize well to data outside of the training set.
training_samples = []

for t in range(30):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print('action_taken_', t, ' :',action)
    h_inp = get_human_feedback()
    training_samples.append([[observation, action], h_inp])
    if terminated:
        print('failed time')
        observation = env.reset()


# Split the data
train_samples, val_samples = train_test_split(training_samples, test_size=0.2, random_state=42)

# Create separate datasets for training and validation
train_data = reward_dataset(train_samples)
val_data = reward_dataset(val_samples)


# Training loop with early stopping
best_val_loss = float('inf')
patience = 5
counter = 0
max_gap = 0.1  # Maximum allowed gap between training and validation loss

for epoch in range(EPOCHS):
    train_loss = train_epoch(RM, train_data, RM_optimizer)
    
    # The validation data is used only to monitor the model's performance on unseen data and to implement early stopping.
    # It does not directly influence the model parameters.
    val_loss = validate(RM, val_data)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    
    loss_gap = train_loss - val_loss
    
    if counter >= patience and loss_gap > max_gap:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("Training completed.")
#torch.save(RM, "reward_model.pkl")
torch.save(RM.state_dict(), "reward_model.pkl")

