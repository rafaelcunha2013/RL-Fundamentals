import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """"The 10-armed Testbed
    It uses fixed learning rate by default
    Choose average=True to an averaging learning rate"""
    def __init__(self, k, alpha=0.1, random_walk=False):
        self.q = [np.random.normal(loc=0, scale=1) for _ in range(k)]
        self.k = k
        self.lr = alpha
        self.random_walk = random_walk

    def reward(self, action):
        return np.random.normal(loc=self.q[action], scale=1)

    def train(self, episodes, epsilon, average=False, fixed=False):
        reward_history = []
        action_history = []
        q = np.zeros(self.k, dtype=float)
        n = np.zeros(self.k, dtype=float)
        for _ in range(episodes):
            if self.random_walk:
                self.q = [self.q[i] + np.random.normal(loc=0, scale=0.01) for i in range(self.k)]
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q)
            else:
                action = np.random.randint(0, 10)
            action_history.append(action)
            reward = self.reward(action)
            reward_history.append(reward)
            n[action] += 1
            if average:
                self.lr = n[action]
            if fixed:
                pass
            q[action] += self.lr * (reward - q[action])
        return reward_history # , q, action_history


def bandit_arm_problem(k, episodes, run):
    bandit = [Bandit(k) for _ in range(run)]

    for epsilon in [0, 0.1, 0.01]:
        reward = [bandit[i].train(episodes, epsilon, average=True) for i in range(run)]
        reward_aux = [list(x) for x in zip(*reward)]
        mean_reward = [np.mean(reward_aux[i]) for i in range(episodes)]
        ax = plt.subplot(111)
        ax.plot(mean_reward, label=epsilon)
        ax.legend()
    plt.show()


def bandit_arm_problem_random_walk(k, episodes, run):
    bandit = [Bandit(k, random_walk=True) for _ in range(run)]
    epsilon = 0.1
    reward_fixed = [bandit[i].train(episodes, epsilon, fixed=True) for i in range(run)]
    reward_average = [bandit[i].train(episodes, epsilon, average=True) for i in range(run)]

    ax = plt.subplot(111)

    reward = reward_fixed
    reward_aux = [list(x) for x in zip(*reward)]
    mean_reward = [np.mean(reward_aux[i]) for i in range(episodes)]
    ax.plot(mean_reward, label="Fixed")

    reward = reward_average
    reward_aux = [list(x) for x in zip(*reward)]
    mean_reward = [np.mean(reward_aux[i]) for i in range(episodes)]
    ax.plot(mean_reward, label="Average")

    ax.legend()
    plt.show()


if __name__ == '__main__':
    k = 10
    episodes = 10000
    run = 2000

    # bandit_arm_problem(k, episodes, run)
    bandit_arm_problem_random_walk(k, episodes, run)




