import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-shape", nargs="+", type=int, required=True)
parser.add_argument("-o", "--output-shape", nargs="+", type=int, required=True)
parser.add_argument("-f", "--path", type=str, required=True)
#TODO: support other output types (needs AegisEnv modification)
parser.add_argument("-d", "--discrete", type=bool, default=False)
parser.add_argument("-l", "--policy", type=str, default="MlpPolicy")
#TODO: more args
args = parser.parse_args()

#TODO: when loading, log path not saved (https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html)
#TODO: specify verbose when loading (and environment..)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import gym
import numpy as np
from utils import DummyEnv

#TODO: support more obs spaces
obs_space = gym.spaces.Box(shape=args.input_shape, low=-np.Inf, high=np.Inf)
#TODO: support more action spaces
if args.discrete:
  action_space = gym.spaces.Discrete(args.output_shape[0])
else:
  action_space = gym.spaces.Box(shape=args.output_shape, low=-np.Inf, high=np.Inf)

env = DummyEnv(obs_space, action_space)
env = DummyVecEnv([lambda: env])

#TODO: infer space types and shapes from saved agent?
#TODO: hardcoded policy
model = PPO2(args.policy, env, verbose=1, nminibatches=1)
model.save(args.path)
