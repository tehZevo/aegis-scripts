import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--urls", nargs="+", required=True)
parser.add_argument('-p','--port', type=str, required=True)
parser.add_argument('-a','--app-name', default="mobilenet")
parser.add_argument('-s','--niceness', type=float, default=1)
parser.add_argument('-o','--pooling', default="max")
parser.add_argument('-r','--resize-to', nargs=2, type=int, default=None)

args = parser.parse_args()

import logging
import matplotlib

#tensorboard logging stuff
import tensorflow as tf
from datetime import datetime
import time

from aegis_core.flask_controller import FlaskController
from aegis_core.keras_app_engine import KerasAppEngine
tf.enable_eager_execution()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

matplotlib.use("Agg") #threading issue

engine = KerasAppEngine(app_name=args.app_name, pooling=args.pooling,
  input_urls=args.urls)

print(engine.input_shape, engine.output_shape)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
