import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--node-urls", type=str, nargs="*", required=False, default=[])
parser.add_argument('-p','--port', required=True)
parser.add_argument("-s", "--niceness", type=float, default=1)
#TODO: scaling, channels, etc

import logging
import tensorflow as tf
import matplotlib

from aegis_core.reward_proxy import RewardProxy
from aegis_core.flask_controller import FlaskController

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

args = parser.parse_args()

engine = RewardProxy(args.node_urls)
controller = FlaskController(engine, port=args.port, niceness=args.niceness)
