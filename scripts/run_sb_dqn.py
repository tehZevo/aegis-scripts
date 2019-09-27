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
#TODO: lr and lots of other DQN params
args = parser.parse_args()

import logging
from stable_baselines import DQN
from aegis_core.aegis_env import AegisEnv

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#TODO: use callback to save
#TODO: train for infinite steps
#TODO: use callback to sleep? :^)
#TODO: attempt to load from path, else create new

timesteps=1e10 #TODO: hardcoded timesteps.. should be infinity

# Create environment
env = AegisEnv(args.input_size, args.output_size, args.urls, port=args.port,
  discrete=args.discrete, sleep=args.sleep)

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1) #1e-3
# Train the agent
model.learn(total_timesteps=int(timesteps), log_interval=1)

#TODO: saving
# Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading
#
# # Load the trained agent
# model = DQN.load("dqn_lunar")
#
# # Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   env.render()
