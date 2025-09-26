import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_true = np.random.rand(n_arms)  # True values of each arm
        self.q_estimates = np.zeros(n_arms)  # Estimated values of each arm
        self.action_counts = np.zeros(n_arms)  # Number of times each arm was selected

    def pull_arm(self, arm):
        # Simulate pulling the arm by returning a reward from a normal distribution
        return np.random.randn() + self.q_true[arm]

    def update_estimates(self, arm, reward):
        self.action_counts[arm] += 1
        self.q_estimates[arm] += (reward - self.q_estimates[arm]) / self.action_counts[arm]

class EpsilonGreedyAgent:
    def __init__(self, bandit, epsilon):
        self.bandit = bandit
        self.epsilon = epsilon

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.bandit.n_arms)  # Explore
        else:
            return np.argmax(self.bandit.q_estimates)  # Exploit

def simulate(bandit, agent, n_steps):
    rewards = np.zeros(n_steps)
    for step in range(n_steps):
        arm = agent.select_arm()
        reward = bandit.pull_arm(arm)
        bandit.update_estimates(arm, reward)
        rewards[step] = reward
    return rewards

# Parameters
n_arms = 10
n_steps = 1000
epsilon = 0.1

# Create bandit and agent
bandit = Bandit(n_arms)
agent = EpsilonGreedyAgent(bandit, epsilon)

# Run simulation
rewards = simulate(bandit, agent, n_steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(rewards) / (np.arange(n_steps) + 1), label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Multi-Armed Bandit - ε-Greedy Strategy')
plt.legend()
plt.grid()
plt.show()