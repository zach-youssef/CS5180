#!/usr/bin/python3

import gymnasium as gym
import numpy as np
from setup import setup

def policy_iteration(P, gamma):
    (env, p, r, S, A) = P

    # Safe access our sparse reward / transition functions
    pf = lambda s, a, sp: p[(s, a, sp)] if (s, a, sp) in p else 0
    rf = lambda s, a, sp: r[(s, a, sp)] if (s, a, sp) in r else 0

    # Initialize policy
    pi = {}
    for s in range(S):
        pi[s] = 0
    pi_next = {}
    k = 0

    V_pi = None
    P_pi = None
    R_pi = None

    while(True):
        # Construct matrices
        P_pi = np.zeros((S, S), dtype=float)
        R_pi = np.zeros(S, dtype=float)
        for s in range(S):
            a = pi[s]
            for s_prime in range(S):
                prob = pf(s, a, s_prime)
                P_pi[s_prime][s] = prob
                R_pi[s] += prob * rf(s, a, s_prime)

        # Solve for V_pi
        V_pi = np.linalg.solve(np.identity(S) - (P_pi * gamma), R_pi)

        # Improve policy
        for s in range(S):
            best_value = 0
            best_action = 0
            for a in range(A):
                value = sum([pf(s, a, s_prime) * (rf(s, a, s_prime) + gamma * V_pi[s_prime]) for s_prime in range(S)])
                if value > best_value:
                    best_value = value
                    best_action = a
            pi_next[s] = best_action

        # Increment iterations and check if we have finished
        k += 1
        if pi == pi_next:
            break
        pi = pi_next.copy()

    return (V_pi, pi_next, k)

if __name__ == "__main__":
    print(policy_iteration(setup(), 0.99))
