import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
#TODO: turn these back into lists (support in AegisEnv)
parser.add_argument("-i", "--input-shape", type=int, required=True)
parser.add_argument("-o", "--output-shape", type=int, required=True)
parser.add_argument('-p','--port', required=True)
#TODO: support other output types (needs AegisEnv modification)
parser.add_argument("-d", "--discrete", type=bool, default=False)
#TODO: make aegisenv support niceness sleep (fix negative)
parser.add_argument("-s", "--sleep", type=float, default=0.1)
parser.add_argument("-e", "--steps", type=int, default=10000) #steps per episode

parser.add_argument("-f", "--path", type=str, required=True)
#TODO: support more algos
parser.add_argument("-a", "--algorithm", type=str, default="ppo2")
parser.add_argument("-l", "--logdir", type=str, default=None)
parser.add_argument("-v", "--verbose", type=int, default=1)
parser.add_argument("-n", "--name", type=str, default=None)
#parser.add_argument('--train', dest='train', action='store_true')
#parser.add_argument('--no-train', dest='train', action='store_false')
#parser.set_defaults(train=True)
args = parser.parse_args()

import logging
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from aegis_core.aegis_env import AegisEnv

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Create environment
env = AegisEnv(args.input_shape, args.output_shape, args.urls, port=args.port,
  discrete=args.discrete, sleep=args.sleep, n_steps=args.steps)
env = DummyVecEnv([lambda: env])

#load model
model = PPO2.load(args.path, env, verbose=args.verbose, tensorboard_log=args.logdir)

#train
ep_counter = 0
while True:
  env.reset() #TODO: is this necessary?
  model.learn(total_timesteps=args.steps, reset_num_timesteps=False, tb_log_name=args.name)
  ep_counter += 1
  #TODO: actual step counter might be off because .learn might have different intervals
  print("Steps: {}".format(ep_counter * args.steps))
  model.save(args.path)
