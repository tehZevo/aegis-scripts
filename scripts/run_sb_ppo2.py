import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
parser.add_argument("-i", "--input-size", type=int, required=True)
parser.add_argument("-o", "--output-size", type=int, required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument("-d", "--discrete", type=bool, default=False)
parser.add_argument("-s", "--sleep", type=float, default=0.1)
#parser.add_argument('--train', dest='train', action='store_true')
#parser.add_argument('--no-train', dest='train', action='store_false')
#parser.set_defaults(train=True)
#TODO: more params
args = parser.parse_args()

import logging
from stable_baselines import DQN
from aegis_core.aegis_env import AegisEnv

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

timesteps=1e10 #TODO: hardcoded timesteps.. should be infinity

# Create environment
env = AegisEnv(args.input_size, args.output_size, args.urls, port=args.port,
  discrete=args.discrete, sleep=args.sleep)

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

vec_env = DummyVecEnv([lambda: env])

model = PPO2("MlpLstmPolicy", vec_env, verbose=1, nminibatches=1)
model.learn(total_timesteps=int(timesteps))
