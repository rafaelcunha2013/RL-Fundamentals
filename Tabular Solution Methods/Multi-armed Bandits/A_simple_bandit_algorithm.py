import numpy as np
#from Bandit_arm import Bandit

"""A simple bandit algorithm
Chapter 2.4, pag. 32, Reinforcement Learning: A introduction, Sutton & Bartto.
Implementation: Rafael F. Cunha"""


class Bandit:
    """"The k-armed bandit"""
    def __init__(self, k, alpha=0.1, random_walk=False):
        self.q = [np.random.normal(loc=0, scale=1) for _ in range(k)]
        self.k = k
        self.lr = alpha
        self.random_walk = random_walk

    def reward(self, action):
        return np.random.normal(loc=self.q[action], scale=1)


# Initialization
def simple_bandit_algorithm(k):
    bandit = Bandit(k)
    q = np.zeros(k, dtype=float)
    n = np.zeros(k, dtype=float)
    epsilon = 0.3
    for _ in range(1000000):
        if np.random.uniform() > epsilon:
            a = q.argmax()
        else:
            a = np.random.randint(0, k)
        r = bandit.reward(a)
        n[a] += 1
        q[a] += (r - q[a])/n[a]
    return q, bandit


if __name__ == '__main__':
    k = 10
    q, bandit = simple_bandit_algorithm(k)
    print(q)
    print(bandit.q)
    print(np.max(abs(q-bandit.q)))


