import numpy as np
"""Value iteration algorithm for a deterministic policy
input: 
    threshold theta
    State space
    Action space
    reward space
    
It is not working. It is necessary to implement the model
"""
# Generate next state
# Model function p (next_state, reward, state, action
# pi (state)


def model(next_state, reward, state, action):
    return np.random.uniform(0, 1)


def value_iteration(theta, state_space, action_space, reward_space):
    # Initialization
    gamma = 0.9
    aux = np.zeros(action_space)
    aux2 = np.zeros([state_space, action_space])
    V = np.random.uniform(-1, 1, state_space)
    V[state_space-1] = 0

    # Loop part
    while True:
        delta = 0
        for s in range(state_space):
            v = V[s]
            for r in range(reward_space):
                for next_state in range(state_space):
                    for a in range(action_space):
                        aux[a] += model(next_state, r, s, a) *(r + gamma*V[next_state])
            V[s] = aux.max()
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    for s in range(state_space):
        for r in range(reward_space):
            for next_state in range(state_space):
                for a in range(action_space):
                    aux2[s, a] += model(next_state, r, s, a) *(r + gamma*V[next_state])
    pi = aux2.argmax(0)
    return pi


if __name__ == '__main__':
    theta = 0.2
    state_space = 5
    action_space = 5
    reward_space = 2
    pi = value_iteration(theta, state_space, action_space, reward_space)
    print(pi)




