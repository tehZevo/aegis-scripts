import argparse
import retro

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument('-x','--proxy')
parser.add_argument('-n','--name', default="")
parser.add_argument('-s','--niceness', type=float, default=1)
parser.add_argument('-i','--interval', default="done")
parser.add_argument('-k','--action-repeat', type=int, default=1)
parser.add_argument('-e','--environment', default="Pong-Atari2600")
#TODO: implement --no-reward (requires modifying env_engine)
parser.add_argument('-r','--end-reward', type=float, default=0)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=False)
parser.add_argument('--state', default=retro.State.DEFAULT)
parser.add_argument('--scenario', default=None)
parser.add_argument('-a','--action-type', type=str, default="filtered", choices=["all", "discrete", "filtered", "multi_discrete"])
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')

#TODO: recording bk2

args = parser.parse_args()

import logging
import matplotlib

#tensorboard logging stuff
import tensorflow as tf
from datetime import datetime

from utils import env_callbacks

tf.enable_eager_execution()
logdir = "./logs/envs/{}".format(args.environment) + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.contrib.summary.create_file_writer(
  logdir, flush_millis=10000)
interval = int(args.interval) if args.interval.isdigit() else args.interval
cbs = env_callbacks(summary_writer, args.environment, interval)
#end logging stuff

from aegis_core.flask_controller import FlaskController
from aegis_core.env_engine import EnvEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

print("Creating {}...".format(args.environment))
obs_type = retro.Observations[args.obs_type.upper()]
action_type = retro.Actions[args.action_type.upper()]
env = retro.make(args.environment, obs_type=obs_type, use_restricted_actions=action_type)
end_reward = args.end_reward

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
#print(env.observation_space.low, env.observation_space.high)

#both ram and image are 0-255 i think
obs_scale = lambda x: x / 255.

engine = EnvEngine(env, end_reward, action_url=args.url, run_name=args.name,
  reward_proxy=args.proxy, action_repeat=args.action_repeat, render=args.render,
  callbacks=cbs, obs_scale=obs_scale)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
