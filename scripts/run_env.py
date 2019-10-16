import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument('-x','--proxy')
parser.add_argument('-n','--name', default="")
parser.add_argument('-s','--niceness', type=float, default=1)
parser.add_argument('-k','--action-repeat', type=int, default=1)
parser.add_argument('-e','--environment', default="CartPole-v0")
#TODO: implement --no-reward (requires modifying env_engine)
parser.add_argument('-r','--end-reward', type=float, default=0)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()

import logging
import gym
import matplotlib

#tensorboard logging stuff
import tensorflow as tf
from datetime import datetime

from utils import env_callbacks

tf.enable_eager_execution()
logdir = "./logs/envs/{}".format(args.environment) + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.contrib.summary.create_file_writer(
  logdir, flush_millis=10000)

cbs = env_callbacks(summary_writer, args.environment)
#end logging stuff

from aegis_core.flask_controller import FlaskController
from aegis_core.env_engine import EnvEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

env = gym.make(args.environment)
end_reward = args.end_reward

#TODO: fix report interval shenanigans..
#TODO: convert everything to callbacks..
# printing "episode X: <reward>"
# saving interval actions (histogram? or fall back to save_plot)
# saving model
#
engine = EnvEngine(env, end_reward, action_url=args.url, run_name=args.name,
  reward_proxy=args.proxy, action_repeat=args.action_repeat, render=args.render,
  callbacks=cbs)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
