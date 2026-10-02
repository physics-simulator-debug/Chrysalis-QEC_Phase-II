import gymnasium as gym 
from gymnasium import spaces 
import numpy as np
import stim

class QuantumSurfaceCodeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, distance=3, noise_rate=0.01, rounds=3):
        super().__init__()
        self.d = distance
        self.noise_rate = noise_rate
        self.rounds = rounds

        self.ancillas_per_round = self.d * (self.d - 1)
        self.total_detectors = self.rounds * self.ancillas_per_round

        self.action_space = spaces.Discrete(self.d * self.d + 1)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.rounds, self.d, self.d - 1), dtype=np.uint8
        )

        self.stim_circuit = self._build_surface_circuit()
        self.sampler = self.stim_circuit.compile_detector_sampler()

    def _build_surface_circuit(self):
        return stim.Circuit.generated(
            "surface_code:unrotated_memory_z",
            distance=self.d,
            rounds=self.rounds,
            after_clifford_depolarization=self.noise_rate
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        detector_samples, actual_observables = self.sampler.sample(shots=1, separate_observables=True)

        self.raw_detectors = np.array(detector_samples[0], dtype=np.uint8)
        self.logical_observable = bool(actual_observables[0][0])

        
        self.agent_flips = np.zeros((self.d, self.d), dtype=np.uint8)

        detectors_needed = self.rounds * self.ancillas_per_round
        clean_detectors = self.raw_detectors[:detectors_needed]
        
        self.state = clean_detectors.reshape((self.rounds, self.d, self.d - 1)).astype(np.uint8)

       
        self.initial_syndrome_count = np.sum(self.state)

        return self.state, {}

    def step(self, action):
        terminated = True 
        truncated = False

        if action > 0:
            target_qubit = action - 1  
            r = target_qubit // self.d
            c = target_qubit % self.d
            self.agent_flips[r, c] = 1

       
        agent_flipped_logical = bool(np.sum(self.agent_flips[0, :]) % 2)

        nature_flipped_logical = self.logical_observable
        final_error = nature_flipped_logical ^ agent_flipped_logical

        if final_error == False:
       
            base_reward = 1.0
        else:
           
            base_reward = -1.5  

        total_flips = np.sum(self.agent_flips)
        
        action_penalty = -0.05 * total_flips
        
        reward = base_reward + action_penalty

        return self.state, reward, terminated, truncated, {}
