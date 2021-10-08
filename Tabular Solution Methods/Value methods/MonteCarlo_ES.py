import numpy as np
from collections import defaultdict
from numpy.random import default_rng
import gym


def generate_episode(env, pi, s0, a0):
    history = []
    state = s0
    action = a0
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        history.append((state, action, reward))
        state = next_state
        action = pi[state]
    return history


def mc_exploring_starts(env, episodes, gamma=1):
    rng = default_rng()
    pi = defaultdict(lambda: rng.integers(env.action_space.n))
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(lambda: [])

    for _ in range(episodes):
        # env.reset() already generates a random initial state
        s0 = env.reset()
        a0 = rng.integers(env.action_space.n)
        history = generate_episode(env, pi, s0, a0)
        g = 0
        states_action = [(s, a) for s, a, _ in history]
        history.reverse()
        for s, a, r in history:
            g = gamma * g + r
            if states_action.pop() not in states_action:
                returns[(s, a)].append(g)
                q[s][a] = np.mean(returns[(s, a)])
                pi[s] = np.argmax(q[s])

    return q, pi


if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    episodes = 100
    q, pi = mc_exploring_starts(env, episodes, gamma=1)
    q_plot = dict((k, v) for k, v in q.items())
    pi_plot = dict((k, v) for k, v in pi.items())
    print(q)
    print(pi)

