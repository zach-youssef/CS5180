#!/usr/bin/python3

from setup import setup
from vi import value_iteration

def find_k_star(P, gamma, theta):
    # Find optimal policy
    _, pi_star, k = value_iteration(P, gamma, theta)
    # Find where the policy converges
    vk, dist, kstar = value_iteration(P, gamma, theta, known_pi=pi_star)
    print(f"pi* converged after {kstar}/{k} iterations")
    print(f"||V(k*)-V(k*-1)|| = {dist}\nVk* = {vk}")

if __name__ == "__main__":
    find_k_star(setup(), 0.99, 10e-4)
