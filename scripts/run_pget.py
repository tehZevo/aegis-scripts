import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
parser.add_argument("-m", "--model-path", type=str, required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-t", "--action_type", type=str, default="continuous")
parser.add_argument("-r", "--reward-propagation", type=float, default=0)
parser.add_argument("-s", "--niceness", type=float, default=1)
parser.add_argument("-e", "--noise", type=float, default=0.1)
parser.add_argument("-c", "--advantage_clip", type=float, default=1.0) #TODO: allow "None"
parser.add_argument("-g", "--gamma", type=float, default=(1-1e-5))
parser.add_argument("-l", "--lambda", dest="lambda_", type=float, default=0.9)
parser.add_argument("-d", "--initial-deviation", type=float, default=10)
parser.add_argument("-k", "--weight-decay", type=float, default=1e-6)
parser.add_argument("-q", "--late-squash", type=bool, default=False)
parser.add_argument('--alt-trace', dest='alttrace', action='store_true')
parser.add_argument('--no-alt-trace', dest='alttrace', action='store_false')
parser.set_defaults(alttrace=False)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('--square-dev', dest='squaredev', action='store_true')
parser.add_argument('--no-square-devtrain', dest='squaredev', action='store_false')
parser.set_defaults(squaredev=False)

import logging
import tensorflow as tf
import matplotlib
import numpy as np

from aegis_core.flask_controller import FlaskController
from aegis_core.rl_engine import RLEngine

from pget import Agent

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

tf.enable_eager_execution()
args = parser.parse_args()

model = tf.keras.models.load_model(args.model_path)

#tensorboard logging stuff
from datetime import datetime
from utils import pget_callbacks
from aegis_core.callbacks import ModelSaver

name = args.name if args.name is not None else args.port
logdir = "./logs/pget/{}-".format(name) + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.contrib.summary.create_file_writer(
  logdir, flush_millis=10000)

cbs = pget_callbacks(summary_writer, name, interval=100, outlier_z=np.Infinity)
#TODO: dont save if not train
cbs.append(ModelSaver(model, args.model_path))
#end logging stuff

agent = Agent(model, action_type=args.action_type, regularization=args.weight_decay,
  alt_trace_method=args.alttrace, advantage_clip=args.advantage_clip,
  lambda_=args.lambda_, gamma=args.gamma, noise=args.noise, optimizer=model.optimizer,
  initial_deviation=args.initial_deviation, late_squash=args.late_squash,
  use_squared_deviation=args.squaredev)

engine = RLEngine(agent, input_urls=args.urls, train=args.train,
  reward_propagation=args.reward_propagation, callbacks=cbs)

controller = FlaskController(engine, port=args.port, niceness=args.niceness)
