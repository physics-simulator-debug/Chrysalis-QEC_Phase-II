import gymnasium as gym
from stable_baselines3 import PPO
from qec_env import QuantumSurfaceCodeEnv  
import numpy as np

#Enter the name of the modle and the Noise rate to start the test
name = 'Chrysalis_S_d3_0_02_C'  
noise = 0.01


distance = 3   
rounds = 3

def evaluate_model():
    print("==================================================")
    print(f"[+] Loading the trained model '{name}' for evaluation...")
    print("==================================================")
    
    #Tern on the Env
    env = QuantumSurfaceCodeEnv(distance=distance, noise_rate=noise, rounds=rounds)
    

    model = PPO.load(name)
    
    num_episodes = 30
    successful_corrections = 0
    
    print(f"\n[+] Testing the model over {num_episodes} episodes:\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        action, _states = model.predict(obs, deterministic=True)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        success = "SUCCESS (Corrected)" if reward > 0 else "FAILED (Uncorrected)"
        if reward > 0:
            successful_corrections += 1
            
        print(f"Episode {episode + 1}:")

        print(f"  - Detector Tensor (Obs):\n{obs}")
        print(f"  - Chosen Action by Agent: {action}")
        print(f"  - Quantum Reward: {reward} -> {success}")
        print("-" * 40)
        
    success_rate = (successful_corrections / num_episodes) * 100
    print("==================================================")
    print(f"[+] Evaluation finished! {name} on noise {noise}")
    print(f"[+] Total Success Rate: {success_rate}%")
    print("==================================================")

    return success_rate

if __name__ == "__main__":
    result = []

    for i in range(10):
        print(f"=========================[{i+1}]================================")
        acc = evaluate_model()
        result.append(acc)
    print("\n================ FINAL RESULTS ================")
    print(f"[+] Model Evaluation on Surface Code finished!")
    print(f"Mean Accuracy (MAE) --->  {np.mean(result)}%")
    print(f"Standard Deviation (STD) ---->  +-{np.std(result)}")
    print("===============================================")
