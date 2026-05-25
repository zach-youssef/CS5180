from env import AppartmentEnv
from numpy import random
from numpy import array

# Dummy interface
class Policy:
    def act(self, obs) -> int:
        pass

class RandomPolicy(Policy):
    def __init__(self, T: int):
        self.T = T

    # Accept 1/T of the time
    def act(self, obs) -> int:
        r = random.randint(0, self.T)
        if r == 0:
            return 1 # accept
        else:
            return 0 # reject

class ThresholdPolicy(Policy):
    def __init__(self, u_min: int):
        self.u_min = u_min;

    # Accept if U_t >= U_min
    def act(self, obs) -> int:
        if obs["ut"] >= self.u_min:
            return 1 # accept
        else:
            return 0 # reject

class OptimalPolicy(Policy):
    def __init__(self):
        self.policy = array([[0, 0, 0, 1],
                             [0, 0, 1, 1],
                             [0, 0, 1, 1],
                             [1, 1, 1, 1]])

    # Returns action based on computed lookup table
    def act(self, obs) -> int:
        t_idx = obs["t"] - 1
        u_idx = obs["ut"] - 1
        # If there's noise, coerce the observed ut into a valid index
        u_idx = max(0, min(3, int(u_idx)))
        return self.policy[t_idx][u_idx]
