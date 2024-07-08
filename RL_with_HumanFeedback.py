import numpy as np

class GridWorldPOMDP:
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
        self.state = (0, 0)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        self.noise = 0.1  # Observation noise level

    def reset(self):
        self.state = (0, 0)
        return self.observe()

    def step(self, action):
        if action in self.actions:
            move = self.action_map[action]
            new_state = (self.state[0] + move[0], self.state[1] + move[1])
            new_state = (
                max(0, min(self.size - 1, new_state[0])),
                max(0, min(self.size - 1, new_state[1]))
            )
            self.state = new_state

        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.observe(), reward, done

    def observe(self):
        noisy_observation = (
            self.state[0] + np.random.normal(0, self.noise),
            self.state[1] + np.random.normal(0, self.noise)
        )
        return noisy_observation

    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.state] = 1
        print(grid)

# Example usage:
env = GridWorldPOMDP()
state = env.reset()
env.render()

def human_feedback(state, action):
    # Simulate human feedback based on some heuristic or knowledge
    # For this example, assume human feedback is perfect (goal-directed)
    correct_action = {
        (0, 0): 'right',
        (0, 1): 'right',
        (0, 2): 'right',
        (0, 3): 'down',
        (1, 3): 'down',
        (2, 3): 'down',
        (3, 3): 'down',
        (4, 3): 'right'
    }
    if correct_action.get(state) == action:
        return 1
    else:
        return -1
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, beta=0.5):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta  # weight for human feedback
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state, feedback):
        action_index = self.actions.index(action)
        q_predict = self.q_table[state][action_index]
        q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.alpha * (q_target - q_predict) + self.beta * feedback

# Example usage:
agent = QLearningAgent(env.actions)

num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    state = tuple(map(int, state))  # Convert continuous observation to discrete
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        next_state = tuple(map(int, next_state))  # Convert continuous observation to discrete
        feedback = human_feedback(state, action)
        agent.learn(state, action, reward, next_state, feedback)
        state = next_state

    if episode % 100 == 0:
        print(f'Episode {episode} complete')

print('Training complete')  

