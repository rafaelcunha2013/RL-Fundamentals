"""
Implementation of the First-visit MC prediction, for estimating V = v_pi
Chapter 5.1, Monte Carlo prediction, Sutton and Barton
By Rafael Fernandes Cunha
October 1, 2021
"""

import numpy as np

def generate_policy(state_space, action_space):
    pi = np.random.uniform(0, 1, [state_space, action_space])
    return pi


def first_visit_mc(pi):
    value = np.random.uniform(-1, 1, pi.shape[0])
    return = [None] * pi.shape[0]
    done = False
    while not done:

        next_state, reward, done, info = env.step(action)

