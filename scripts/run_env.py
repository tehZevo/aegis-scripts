import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument('-n','--name', required=False)
parser.add_argument('-s','--niceness', required=False, default=1)
parser.add_argument('-e','--environment', required=False, default="CartPole-v0")
parser.add_argument('-r','--end-reward', required=False, type=float, default=0)

args = parser.parse_args()

import logging
import gym
import matplotlib

from aegis_core.flask_controller import FlaskController
from aegis_core.env_engine import EnvEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

#TODO: define env and end of episode reward

env = gym.make(args.environment)
end_reward = args.end_reward

engine = EnvEngine(env, end_reward, action_url=args.url, run_name=args.name)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
