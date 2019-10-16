import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--input-urls", nargs="+", required=True)
parser.add_argument("-a", "--action-url", required=True)
parser.add_argument("-m", "--model-path", type=str, required=True)
parser.add_argument('-p','--port', required=True)
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-s", "--niceness", type=float, default=1)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
#TODO: other args

import logging
import tensorflow as tf
import matplotlib

from aegis_core.flask_controller import FlaskController
from aegis_core.curiosity import LocalCuriosityEngine

from pget import Agent

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

tf.enable_eager_execution()
args = parser.parse_args()

#tensorboard logging stuff
from datetime import datetime
from utils import curiosity_callbacks

name = args.name if args.name is not None else args.port
logdir = "./logs/curiosity/{}-".format(name) + datetime.now().strftime("%Y%m%d-%H%M%S")

summary_writer = tf.contrib.summary.create_file_writer(
  logdir, flush_millis=10000)

cbs = curiosity_callbacks(summary_writer, args.name, interval=100)
#end logging stuff

model = tf.keras.models.load_model(args.model_path)
#TODO: other args
engine = LocalCuriosityEngine(model, args.action_url, args.input_urls,
  train=args.train, callbacks=cbs)

controller = FlaskController(engine, port=args.port, niceness=args.niceness)
