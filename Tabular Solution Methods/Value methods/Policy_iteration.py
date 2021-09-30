import numpy as np
import random
"""
Iterative Policy Evaluation, for estimating V ~ v_pi
"""


def compute_next_state(state, action):
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


def p_dynamics(next_state, r, state, action):
    result = 0
    if next_state == compute_next_state(state, action) and r == -1:
        result = 1
    return result


def policy_evaluation(pi, V, gamma):
    theta = 0.001
    k = len(V)
    r = -1
    while True:
        delta = 0
        for state in range(1, k-1):
            v = V[state]
            # V[state] = r + gamma * V[compute_next_state(state, pi[state]]   # == This line also works,  less general
            V[state] = np.sum([p_dynamics(next_state, r, state, pi[state]) *
                               (r + gamma * V[next_state]) for next_state in range(k)])
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V


def policy_improvement(pi, V, A, gamma):
    r = -1
    k = len(V)
    policy_stable = True
    for state in range(1, k-1):
        old_action = pi[state]
        pi[state] = A[np.argmax([r + gamma*V[compute_next_state(state, action)] for action in A])]
        if old_action != pi[state]:
            policy_stable = False
    return pi, policy_stable


def policy_iteration(A):
    V = np.zeros(16, dtype=float)
    det_pi = [random.choice(A) for _ in range(16)]
    gamma = 0.9
    while True:
        V = policy_evaluation(det_pi, V, gamma)
        det_pi, policy_stable = policy_improvement(det_pi, V, A, gamma)
        if policy_stable:
            break
    return V, det_pi


A = ['up', 'down', 'right', 'left']
# pi = np.random.dirichlet(np.ones(len(A)), size=len(S))
# pi = np.ones([14, 4]) * 0.25
# iterative_policy_evaluation(pi, A)
# V = np.zeros(16, dtype=float)

V, pi = policy_iteration(A)
np.set_printoptions(precision=4)
print(V.reshape(4, 4))
print(np.array(pi).reshape(4, 4))

