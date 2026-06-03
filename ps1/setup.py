#!/usr/bin/python3

import gymnasium as gym

# Returns (env, p, r, S, A)
# p and r are maps with (s, a, s') keys
# s is the number of states
# a is the number of actions
def setup():
    env =  gym.make("FrozenLake-v1", is_slippery=True)
    p = {}
    r = {}
    S = env.observation_space.n
    A = env.action_space.n
    for s in range(S):
        for a in range(A):
            entries = env.unwrapped.P[s][a]
            for (prob, next_state, reward, terminated) in entries:
                p[(s, a, next_state)] = prob
                r[(s, a, next_state)] = reward

    return (env, p, r, S, A)

if __name__ == '__main__':
    env, p, r, S, A = setup()
    print(S, A)
