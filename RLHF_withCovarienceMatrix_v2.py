import numpy as np
from scipy.stats import multivariate_normal

class POMDPWithCovariance:
    def __init__(self, size=2):
        self.size = size
        self.true_state = np.random.rand(size)
        self.belief_mean = np.zeros(size)
        self.belief_cov = np.eye(size)  # Initial covariance matrix

    def observe(self):
        # Measurement model with noise
        measurement_noise_cov = np.eye(self.size) * 0.1
        return multivariate_normal.rvs(mean=self.true_state, cov=measurement_noise_cov)

    def update_belief(self, action, observation):
        # Prediction step
        predicted_mean = self.belief_mean + action
        process_noise_cov = np.eye(self.size) * 0.05
        predicted_cov = self.belief_cov + process_noise_cov

        # Update step (using Kalman filter equations)
        measurement_noise_cov = np.eye(self.size) * 0.1
        kalman_gain = predicted_cov @ np.linalg.inv(predicted_cov + measurement_noise_cov)
        self.belief_mean = predicted_mean + kalman_gain @ (observation - predicted_mean)
        self.belief_cov = (np.eye(self.size) - kalman_gain) @ predicted_cov

    def step(self, action):
        # Action is now a vector of the same size as the state
        self.true_state += action + np.random.normal(0, 0.1, self.size)
        observation = self.observe()
        self.update_belief(action, observation)
        
        # Reward is higher if the belief is more certain (lower trace of covariance matrix)
        reward = 1 / (1 + np.trace(self.belief_cov))
        
        return observation, reward, False, {}

def human_feedback():
    feedback = input("Enter feedback (-1: bad, 0: neutral, 1: good): ")
    return float(feedback)

def train_with_human_feedback(env, episodes=5):
    for episode in range(episodes):
        obs = env.observe()
        total_reward = 0
        
        for step in range(3):  # 3 steps per episode
            print(f"Episode {episode+1}, Step {step+1}")
            print(f"Belief mean: {env.belief_mean}")
            print(f"Belief covariance:\n{env.belief_cov}")
            
            # Simple policy: move towards origin
            action = -env.belief_mean * 0.1
            
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            print(f"Action taken: {action}")
            print(f"Observation: {obs}")
            print(f"Reward: {reward}")
            
            # Get human feedback
            feedback = human_feedback()
            
            # Incorporate feedback to adjust belief
            if feedback != 0:
                feedback_strength = 0.1
                env.belief_mean += feedback * feedback_strength * (obs - env.belief_mean)
                env.belief_cov *= (1 - abs(feedback) * feedback_strength)
            
            print(f"Updated belief mean: {env.belief_mean}")
            print(f"Updated belief covariance:\n{env.belief_cov}")
            print("---")
        
        print(f"Episode {episode+1} total reward: {total_reward}")
        print("=====")

# Run the training
env = POMDPWithCovariance()
train_with_human_feedback(env)