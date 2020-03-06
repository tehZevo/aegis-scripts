import argparse
import logging
from datetime import datetime

import retro
import tensorflow as tf

from atari_gauntlet import AtariGauntlet
from utils import env_callbacks

parser = argparse.ArgumentParser()
#aegis params
parser.add_argument("-u", "--url", required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument('-x','--proxy')
parser.add_argument('-n','--name', default="")
parser.add_argument('-s','--niceness', type=float, default=1)
parser.add_argument('-i','--interval', default="done")
parser.add_argument('-k','--action-repeat', type=int, default=1)
#TODO: implement --no-reward (requires modifying env_engine)
parser.add_argument('-r','--end-reward', type=float, default=0)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=False)

#atari gauntlet params
parser.add_argument('-l', '--step-limit', type=int, default=None)
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument("-g", "--allowed-games", type=str, nargs="+", default=None)

args = parser.parse_args()

#TODO: recording bk2

#tensorboard logging stuff

tf.enable_eager_execution()
logdir = "./logs/envs/atari_gauntlet" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.contrib.summary.create_file_writer(
  logdir, flush_millis=10000)
interval = int(args.interval) if args.interval.isdigit() else args.interval
cbs = env_callbacks(summary_writer, "atari_gauntlet", interval)
#end logging stuff

from aegis_core.flask_controller import FlaskController
from aegis_core.env_engine import EnvEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#TODO: handle this inside AtariGauntlet?
obs_type = retro.Observations[args.obs_type.upper()]
#TODO: remove debug flag
env = AtariGauntlet(args.step_limit, obs_type, allowed_games=args.allowed_games, debug=True)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

#both ram and image are 0-255 i think
obs_scale = lambda x: x / 255.

engine = EnvEngine(env, args.end_reward, action_url=args.url, run_name=args.name,
  reward_proxy=args.proxy, action_repeat=args.action_repeat, render=args.render,
  callbacks=cbs, obs_scale=obs_scale)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
