import gym
import gym_conservation

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Parallel environments
env = make_vec_env("conservation-v5", n_envs=4)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log = "/var/log/tensorboard/millie")

model.learn(total_timesteps=100000)

model.save("conservation-v5-A2C-millie")

eval_env = Monitor(gym.make("conservation-v5"))
score = evaluate_policy(model, Monitor(env), n_eval_episodes=10)
score

