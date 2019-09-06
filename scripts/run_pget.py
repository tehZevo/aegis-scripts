import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument("-d", "--discrete", type=bool, default=False)
parser.add_argument("-s", "--niceness", type=float, default=1)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
#TODO: hyperparams

import logging
import tensorflow as tf
import matplotlib

from aegis.flask_controller import FlaskController
from aegis.pget_engine import PGETEngine

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

tf.enable_eager_execution()
print(tf.executing_eagerly())

args = parser.parse_args()

engine = PGETEngine(args.model, is_discrete=args.discrete, input_urls=args.urls, train=args.train)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
