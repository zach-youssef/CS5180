from env import AppartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy
import matplotlib.pyplot as plt
from math import sqrt

def episode(env, policy):
    state, _info = env.reset()
    done = False
    utility = 0
    while not done:
        action = policy.act(state)
        state, utility, done, _trunc, _info = env.step(action)
    return utility

def run_report(env, policy, hist_axs, n=10_000):
    # Run the policy n times
    utilities = [episode(env,policy) for _i in range(n)]
    
    # Compute mean and all reject %
    mean = 0;
    all_reject = 0;
    for u in utilities:
        mean += u
        if (u == 0):
            all_reject += 1
    mean /= len(utilities)
    all_reject /= len(utilities)

    # Compute stdev 
    square_err = [(u - mean) * (u - mean) for u in utilities]
    variance = sum(square_err) / (len(utilities) - 1)
    sigma = sqrt(variance)

    # Create histogram plot
    hist_axs.hist(utilities)

    return (mean, sigma, all_reject)

if __name__ == "__main__":
    # Set environment parameters
    T = 4
    K = 4

    # Create policies
    policies = [
        ("Random", RandomPolicy(T)),
        ("Threshold_1", ThresholdPolicy(1)),
        ("Threshold_2", ThresholdPolicy(2)),
        ("Threshold_3", ThresholdPolicy(3)),
        ("Threshold_4", ThresholdPolicy(4)),
        ("Optimal", OptimalPolicy())
    ]

    # Initialize our standard deviations for noise
    noise_stds = [0, 0.5, 1.0, 2.0]

    # Setup histogram plots
    fig, axs = plt.subplots(len(noise_stds), 
                            len(policies), 
                            sharey=True, 
                            tight_layout=True)

    # For each noise level
    for j in range(len(noise_stds)):
        # Initialize environment
        env = AppartmentEnv(T, K, noise_std=noise_stds[j])

        print(f"Noise Std {noise_stds[j]}:")

        axs[j][0].set_ylabel(f"Noise STD {noise_stds[j]}")

        # Run & evaluate each policy
        for i in range(len(policies)):
            label, policy = policies[i]
            if j == 0:
                axs[j][i].set_title(label)
            mean, sigma, all_reject = run_report(env, policy, axs[j][i])
            print(f"{label}: \tMean {mean}\tStdev {sigma:.4f}\tAll Reject % {all_reject}")

    plt.show()
