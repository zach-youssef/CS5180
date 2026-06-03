#!/usr/bin/python3

import gymnasium as gym
from setup import setup
from functools import reduce
from math import sqrt

# Helper to comptue distance between value functions
def ssd(a, b, S):
    dist = 0
    for s in range(S):
        diff = a[s] - b[s]
        dist += diff * diff
    return sqrt(dist)

def extract_policy(P, v, gamma):
    (env, p, r, S, A) = P
    # Safe access our sparse reward / transition functions
    pf = lambda s, a, sp: p[(s, a, sp)] if (s, a, sp) in p else 0
    rf = lambda s, a, sp: r[(s, a, sp)] if (s, a, sp) in r else 0

    pi = {}
    for s in range(S):
        best_value = 0
        best_action = 0
        for a in range(A):
            value = sum([pf(s, a, s_prime) * (rf(s, a, s_prime) + gamma * v[s_prime]) for s_prime in range(S)])
            if value > best_value:
                best_value = value
                best_action = a
        pi[s] = best_action
    return pi


def value_iteration(P, gamma, theta, known_pi=None):
    (env, p, r, S, A) = P

    # Safe access our sparse reward / transition functions
    pf = lambda s, a, sp: p[(s, a, sp)] if (s, a, sp) in p else 0
    rf = lambda s, a, sp: r[(s, a, sp)] if (s, a, sp) in r else 0
    
    # initialize v_0(s) = s forall s in S
    v_prev = {}
    for s in range(S):
        v_prev[s] = 0

    v_next = {}
    k = 0

    # Perform value iteration
    while(True):
        for s in range(S):
            v_next[s] = reduce(lambda v0,v1: max(v0,v1), [sum([pf(s, a, s_prime) * (rf(s, a, s_prime) + gamma * v_prev[s_prime]) for s_prime in range(S)]) for a in range(A)])

        k += 1

        if known_pi != None:
            # Check the current policy
            pi = extract_policy(P, v_next, gamma)
            if pi == known_pi:
                return (v_next, ssd(v_prev, v_next, S), k)

        if ssd(v_prev, v_next, S) < theta:
            break

        v_prev = v_next.copy()

    # Exctract greedy policy
    pi_star = extract_policy(P, v_next, gamma)

    # Return (v*, pi*, k)
    return (v_next, pi_star, k)

if __name__ == "__main__":
    print(value_iteration(setup(), 0.99, 10e-4))


