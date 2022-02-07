import gym
import gym_conservation

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = gym.make("conservation-v5")

model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000)

model.save("conservation-v5-A2C-millie")
