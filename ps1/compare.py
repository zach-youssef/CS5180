#!/usr/bin/python3

import matplotlib.pyplot as plt
from datetime import datetime

from pi import policy_iteration
from vi import value_iteration
from setup import setup

# Times the given function and returns (iterations, duration)
def benchmark(f, gamma):
    start = datetime.now();
    _, _, k = f(gamma)
    duration = start - datetime.now()
    return (k, duration.microseconds)

# Returns (vi_iterations, pi_iterations, vi_durations, pi_durations)
def compare(P, gammas):
    p_itr = lambda gamma: policy_iteration(P, gamma)
    v_itr = lambda gamma: value_iteration(P, gamma, 10e-4)

    vi_iterations = []
    pi_iterations = []
    vi_durations = []
    pi_durations = []

    for gamma in gammas:
        k, duration = benchmark(v_itr, gamma)
        vi_iterations.append(k)
        vi_durations.append(duration)

        k, duration = benchmark(p_itr, gamma)
        pi_iterations.append(k)
        pi_durations.append(duration)

    return (vi_iterations, pi_iterations, vi_durations, pi_durations)

# Returns (vi_backups, pi_backups)
def compute_backups(P, vi_iterations, pi_iterations):
    _, _, _, S, A = P
    return ([k * S * S * A for k in vi_iterations], [k * S * S * S for k in pi_iterations])

def plot(gammas, vi_iterations, pi_iterations):
    plt.plot(gammas, vi_iterations, label="Value Iteration")
    plt.plot(gammas, pi_iterations, label="Policy Iteration")
    plt.title("Iteration count vs γ")
    plt.ylabel("γ")
    plt.xlabel("Iteration count")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    P = setup()
    gammas = [0.5, 0.9, 0.99, 0.999]
    vi_iterations, pi_iterations, vi_durations, pi_durations = compare(P, gammas)
    vi_backups, pi_backups = compute_backups(P, vi_iterations, pi_iterations)

    for i in range(len(gammas)):
        print(f"Results for γ = {gammas[i]}:\nMethod\tIterations\tDuration\tBackups\nValueItr\t{vi_iterations[i]}\t{vi_durations[i]}\t\t{vi_backups[i]}\nPolicyItr\t{pi_iterations[i]}\t{pi_durations[i]}\t\t{pi_backups[i]}")

    plot(gammas, vi_iterations, pi_iterations)
