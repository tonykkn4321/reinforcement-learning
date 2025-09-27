import numpy as np
import matplotlib.pyplot as plt

class GradientBandit:
    def __init__(self, n_arms, alpha):
        self.n_arms = n_arms
        self.alpha = alpha  # Step size
        self.preferences = np.zeros(n_arms)  # Preferences for each arm
        self.action_counts = np.zeros(n_arms)  # Number of times each arm was selected
        self.probabilities = np.ones(n_arms) / n_arms  # Action probabilities (softmax)

    def softmax(self):
        """Calculate the action probabilities using softmax"""
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))  # Stability
        self.probabilities = exp_preferences / np.sum(exp_preferences)

    def select_arm(self):
        """Select an arm based on the probabilities"""
        self.softmax()
        return np.random.choice(self.n_arms, p=self.probabilities)

    def pull_arm(self, arm):
        """Simulate pulling the arm and return a reward"""
        # Simulate a reward (for example, Gaussian rewards centered around arm's true value)
        true_rewards = np.random.rand(self.n_arms)  # True reward values for each arm
        return np.random.normal(true_rewards[arm], 1)

    def update_preferences(self, arm, reward):
        """Update the preferences based on the received reward"""
        # Calculate the baseline (average reward)
        average_reward = np.mean(self.action_counts)
        for a in range(self.n_arms):
            if a == arm:
                self.preferences[a] += self.alpha * (reward - average_reward) * (1 - self.probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - average_reward) * self.probabilities[a]

def simulate(bandit, n_steps):
    rewards = np.zeros(n_steps)
    for step in range(n_steps):
        arm = bandit.select_arm()
        reward = bandit.pull_arm(arm)
        bandit.update_preferences(arm, reward)
        rewards[step] = reward
    return rewards

# Parameters
n_arms = 10
n_steps = 1000
alpha = 0.1  # Step size

# Create bandit
bandit = GradientBandit(n_arms, alpha)

# Run simulation
rewards = simulate(bandit, n_steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(rewards) / (np.arange(n_steps) + 1), label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Gradient Bandit Algorithm')
plt.legend()
plt.grid()
plt.show()