import sys
from stable_baselines3 import PPO
from src.env import FloodAllocEnv

env = FloodAllocEnv()  # random scenarios: 3 to 12 zones, random resources
model = PPO("MlpPolicy", env, seed=0, n_steps=2048, batch_size=256, learning_rate=3e-4)
model.learn(total_timesteps=int(sys.argv[1]) if len(sys.argv) > 1 else 300_000)
model.save("models/ppo_alloc")
print("saved models/ppo_alloc")
