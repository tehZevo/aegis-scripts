import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument('-n','--name', required=False)
parser.add_argument('-s','--niceness', required=False, default=1)
#TODO: arg for end_reward (-r)
#TODO: env (-e)

args = parser.parse_args()

import logging
import gym
import matplotlib

from aegis.flask_controller import FlaskController
from aegis.env_engine import EnvEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

#TODO: define env and end of episode reward

env = gym.make("CartPole-v0")
end_reward = -1

engine = EnvEngine(env, end_reward, action_url=args.url, run_name=args.name)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
