import numpy as np
import random


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
    prob = np.zeros([])
    result = 0
    if next_state == compute_next_state(state, action) and r == -1:
        result = 1
    return result


def dynamics(state, action):
    next_state = [0, 0]
    x_in = [np.random.poisson(lam=3), np.random.poisson(lam=2)]
    x_out = [np.random.poisson(lam=3), np.random.poisson(lam=4)]
    next_state[0] = max(0, min(20, state[0] + x_in[0] - x_out[0] + action))
    next_state[1] = max(0, min(20, state[1] + x_in[1] - x_out[1] - action))
    r = 10*(x_out[0]+x_out[1]) - 2 * abs(action)
    return next_state, r


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


if __name__ == '__main__':
    state = [np.random.randint(0, 21), np.random.randint(0, 21)]
    pi = np.zeros([20, 20])
    action = pi[state[0]][state[1]]
    next_state, r = dynamics(state, action)