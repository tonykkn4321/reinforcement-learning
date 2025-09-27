import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_true = np.random.rand(n_arms)  # True values of each arm
        self.q_estimates = np.zeros(n_arms)    # Estimated values of each arm
        self.action_counts = np.zeros(n_arms)  # Number of times each arm was selected

    def pull_arm(self, arm):
        # Simulate pulling the arm with a non-stationary reward
        self.q_true += np.random.normal(0, 0.01, self.n_arms)  # Small random walk
        return np.random.randn() + self.q_true[arm]

    def update_estimates(self, arm, reward):
        self.action_counts[arm] += 1
        self.q_estimates[arm] += (reward - self.q_estimates[arm]) / self.action_counts[arm]

class UCBAgent:
    def __init__(self, bandit):
        self.bandit = bandit
        self.total_count = 0

    def select_arm(self):
        self.total_count += 1
        ucb_values = self.bandit.q_estimates + np.sqrt((2 * np.log(self.total_count)) / (self.bandit.action_counts + 1e-5))
        return np.argmax(ucb_values)

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

# Create bandit and agent
bandit = Bandit(n_arms)
agent = UCBAgent(bandit)

# Run simulation
rewards = simulate(bandit, agent, n_steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(rewards) / (np.arange(n_steps) + 1), label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Non-Stationary Multi-Armed Bandit - UCB Strategy')
plt.legend()
plt.grid()
plt.show()