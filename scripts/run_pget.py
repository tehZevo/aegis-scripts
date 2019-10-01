import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument("-d", "--discrete", type=bool, default=False)
parser.add_argument("-r", "--reward-propagation", type=float, default=0)
parser.add_argument("-s", "--niceness", type=float, default=1)
parser.add_argument("-n", "--noise", type=float, default=0.1)
parser.add_argument("-a", "--alpha", type=float, default=1e-4)
parser.add_argument("-o", "--optimizer", type=str, default="adam")
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
#TODO: hyperparams

import logging
import tensorflow as tf
import matplotlib

from aegis_core.flask_controller import FlaskController
from aegis_core.pget_engine import PGETEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

tf.enable_eager_execution()
print(tf.executing_eagerly())

args = parser.parse_args()

opt = None if args.optimizer == "none" else args.optimizer

#TODO: successes in double/triple agent and double lunar had 0 regularization, set back to 0 if issues arise
engine = PGETEngine(args.model, is_discrete=args.discrete, input_urls=args.urls,
  regularization_scale=1e-9, lr=args.alpha, train=args.train, alt_trace_method=False,
  advantage_clip=1, lambda_=0.9, noise=args.noise, optimizer=opt, reward_propagation=args.reward_propagation)#, gamma=0.99)
#engine.optimizer = None #TODO: remove

controller = FlaskController(engine, port=args.port, niceness=args.niceness)
