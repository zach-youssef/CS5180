from gymnasium import Env
from gymnasium import spaces
import numpy as np

class AppartmentEnv(Env):
    def __init__(self, T: int, K: int, seed=None, noise_std=0):
        self._T = T
        self._K = K
        self._nstd = noise_std;

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "t": spaces.Box(1, T, dtype=int),
                "ut": spaces.Box(1, K, dtype=int, seed=seed),
            }
        )
        self._state = {"t": 0, "ut": None};
        self._realU = None;

    def _sampleU(self):
        self._realU = self.observation_space["ut"].sample()[0]
        self._state["ut"] = self._realU + np.random.normal(0, self._nstd * self._nstd)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set t to first week
        self._state["t"] = 1;

        # Sample U
        self._sampleU()

        # returns (obs, info)
        return (self._state, {})

    # returns (obs, reward, terminated, truncated, info)
    def step(self, action: int):
        # Terminate on accept w/ selected reward
        if action == 1:
            return (self._state, self._realU, True, False, {})

        # If this is the last week and we rejected,
        # terminate with 0 reward
        if self._T == self._state["t"]:
            return (self._state, 0, True, False, {})

        # Otherwise, progress to the next week
        self._state["t"] += 1;
        self._sampleU()
        return (self._state, 0, False, False, ())

    
