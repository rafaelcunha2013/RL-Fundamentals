"""
Implementation of the First-visit MC prediction, for estimating V = v_pi
Chapter 5.1, Monte Carlo prediction, Sutton and Barton
By Rafael Fernandes Cunha
October 1, 2021
"""

import numpy as np
from numpy.random import default_rng

class BlackJack:
    def __init__(self):
        self.action_space.n = 4
        self.state_space.n = 5
        self.done = False
        self.info = {}
        self.state = 1
        pass

    def step(self, action):
        return next_state, reward, self.done, self.info

    def reset(self):
        self.state = 1
        self.done = False
        return self.state

class Agent:
    def __init__(self, pi, env):
        self.policy = pi
        self.env = env

    def decision(self, state):
        probs = self.policy[state]
        rng = default_rng()
        action = rng.choice(np.arange(self.env.action_space.n, p=probs))
        rng.choice
        np.arange
        return action

def generate_episode(env, agent):
    history = []
    done = False
    state = env.reset()
    while not done:
        action = agent.decision(state)
        next_state, reward, done, info = env.step(action)
        history.append((state, action, reward))
        state = next_state
    return history

def generate_policy(env):
    rng = default_rng()
    # alpha is the weights of the dirichlet distribution
    alpha = rng.integers(1, 10, env.action_space.n)
    pi = rng.dirichlet(alpha, env.state_space.n)
    return pi


def first_visit_mc(pi):
    value = np.random.uniform(-1, 1, pi.shape[0])
    return = [None] * pi.shape[0]
    done = False
    while not done:
        action = agent(state)
        next_state, reward, done, info = env.step(action)


