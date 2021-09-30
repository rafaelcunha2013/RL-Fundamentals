import numpy as np
import random
"""
Iterative Policy Evaluation, for estimating V ~ v_pi
"""


def iterative_policy_evaluation(pi, A):
    theta = 0.01
    V = np.zeros(16, dtype=float)
    n = np.shape(pi)[1]
    gamma = 1
    # V = np.random.random([4, 4])
    while True:
        delta = 0
        for state in range(1, 15):
            v = V[state]
            V[state] = np.sum([pi[state-1][i] * (-1 + gamma * V[my_next_state(A[i], state)]) for i in range(n)])
            delta = max(delta, abs(v - V[state]))
        np.set_printoptions(precision=0)
        print(V.reshape(4, 4))
        if delta < theta:
            break
    return V


def my_next_state(action, state):
    x = state % 4
    y = state // 4
    if action == 'up':
        y = max(0, y - 1)
    if action == 'down':
        y = min(3, y + 1)
    if action == 'right':
        x = min(3, x + 1)
    if action == 'left':
        x = max(0, x - 1)
    next_state = x + 4 * y
    return next_state


A = ['up', 'down', 'right', 'left']
# pi = np.random.dirichlet(np.ones(len(A)), size=len(S))
pi = np.ones([14, 4]) * 0.25

iterative_policy_evaluation(pi, A)







