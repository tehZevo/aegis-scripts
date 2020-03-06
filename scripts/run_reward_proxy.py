import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--node-urls", type=str, nargs="*", required=False, default=[])
parser.add_argument("-c", "--channels", type=str, nargs="*", required=False)
parser.add_argument("-d", "--decay-rates", type=float, nargs="*", required=False)
parser.add_argument("-l", "--clips", type=float, nargs="*", required=False)
parser.add_argument("-k", "--scales", type=float, nargs="*", required=False)

parser.add_argument('-p','--port', required=True)
parser.add_argument("-s", "--niceness", type=float, default=1)
#TODO: scaling, channels, etc

import logging

from aegis_core.reward_proxy import RewardProxy

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

args = parser.parse_args()

#handle some messy args
other_args = {}
if args.clips is not None:
  other_args["clips"] = args.clips
if args.decay_rates is not None:
  other_args["decay_rates"] = args.decay_rates
if args.scales is not None:
  other_args["scales"] = args.scales

engine = RewardProxy(args.node_urls, channels=args.channels, port=args.port, niceness=args.niceness, **other_args)
