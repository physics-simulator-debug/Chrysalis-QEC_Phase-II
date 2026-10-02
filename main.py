# main.py
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from qec_env import QuantumSurfaceCodeEnv  

class CustomCNNExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        num_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


def train_from_scratch():
    print("==============================================================")
    print("[+] Start up the model training from SCRATCH for SURFACE CODE")
    print("==============================================================")
    
    current_noise = 0.005 
    distance = 3  
    rounds = 3    
    
    env = QuantumSurfaceCodeEnv(distance=distance, noise_rate=current_noise, rounds=rounds)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    model = PPO(
        "CnnPolicy",  
        env,
        verbose=1,
        learning_rate=0.0003, 
        n_steps=2048,          
        batch_size=128,        
        n_epochs=10,
        ent_coef=0.02,         
        policy_kwargs=policy_kwargs 
    )
    
    print(f"[+] Training the baseline model on noise ---> {current_noise * 100}%...")
    
    
    model.learn(total_timesteps=50000)  
    
    model_name = "Chrysalis_S_d3_0_005"
    model.save(model_name)
    
    print(f"\n[+] Baseline model saved successfully ---> '{model_name}.zip'")
    print("==================================================")

if __name__ == "__main__":
    train_from_scratch()
