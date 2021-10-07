"""
Implementation of the First-visit MC prediction, for estimating V = v_pi
Chapter 5.1, Monte Carlo prediction, Sutton and Barton
By Rafael Fernandes Cunha
October 1, 2021
"""

import numpy as np
from numpy.random import default_rng
import gym
from collections import defaultdict
import pandas as pd


class BlackJack(gym.Env):
    def __init__(self):
        self.action_space.n = 4
        self.observation_space.n = 5
        self.done = False
        self.info = {}
        self.state = 1
        pass

    def step(self, action):
        next_state = env.observation_space #np.random.uniform(0, 10)
        reward = np.random.randint(0, 2)
        if np.random.randint(0, 15) == 0:
            self.done = True
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
        probs = self.policy[state[0]]
        rng = default_rng()
        action = rng.choice(np.arange(self.env.action_space.n), p=probs)
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
    pi = rng.dirichlet(alpha, env.observation_space[0].n)
    return pi


def first_visit_mc(env, agent, episodes):
    value = defaultdict(lambda: 0)
    returns = defaultdict(lambda: [])
    gamma = 0.9

    for _ in range(episodes):
        history = generate_episode(env, agent)
        g = 0
        states = [s for s, _, _ in history]
        history.reverse()
        for s, a, r in history:
            g = gamma * g + r
            if states.pop() not in states:
                returns[s].append(g)
                value[s] = np.mean(returns[s])
    return value



if __name__ == "__main__":
    # env = BlackJack
    env = gym.make('Blackjack-v1')
    pi = generate_policy(env)
    agent = Agent(pi, env)
    history = generate_episode(env, agent)
    value = first_visit_mc(env, agent, 4000)
    df = pd.DataFrame(list(value.items()), columns = ['State', 'value'])




