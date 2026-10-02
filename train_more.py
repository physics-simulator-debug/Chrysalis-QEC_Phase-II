import gymnasium as gym
from stable_baselines3 import PPO
from qec_env import QuantumSurfaceCodeEnv  

#A function to controle the learning rate 
def get_linear_lr(initial_lr):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule

def train_more_stages():
    print("=========================================================")
    print("[+] Start up the Upgrade model training for SURFACE CODE")
    print("=========================================================")
    
    new_noise = 0.02 
    distance = 3        
    rounds = 3           
    

    evolve_timesteps = 100000  
    fine_tune_lr = 0.0003       
    
    env = QuantumSurfaceCodeEnv(distance=distance, noise_rate=new_noise, rounds=rounds)
    
    input_model_name = "Chrysalis_S_d3_0_01_C" # Entert the modle name to start retraining it 
    
    print(f"[+] Loading previous expertise from: '{input_model_name}.zip'")
    
    model = PPO.load(input_model_name, env=env)
    
    model.learning_rate = get_linear_lr(fine_tune_lr)
    
    print(f"[+] Continuing training on new noise rate --> {new_noise * 100}%...")
    print(f"[+] Surface Code Setup: Distance={distance}, Rounds={rounds}")
    print(f"[+] Assigned Timesteps for this expansion: {evolve_timesteps}")
    
    
    model.learn(total_timesteps=evolve_timesteps, reset_num_timesteps=False) 
    
   
    output_model_name = "Chrysalis_S_d3_0_02_C"  #Save the new model
    model.save(output_model_name)
    
    print(f"\n[+] WE saved the upgraded model with name ---> '{output_model_name}.zip'")
    print("==================================================")

if __name__ == "__main__":
    train_more_stages()
