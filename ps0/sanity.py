from env import AppartmentEnv 
from numpy import random

if __name__ == "__main__":
    T = 4;
    K = 4;

    env = AppartmentEnv(T, K)
    env.reset()
    
    done = False
    while not done:
        # Choose random action
        action = random.randint(0,2)

        # Step
        obs, reward, terminated, _truncated, _info = env.step(action)

        # Print
        print("({}, {}, {}, {}, {})".format(obs["t"], obs["ut"], action, reward, terminated))

        # Update loop
        done = terminated

