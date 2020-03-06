import logging
import subprocess
import time

from aegis_core.reward_proxy import RewardProxy

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

delay = 1
port = 6000

def allocate():
  global port
  #TODO: check if port is open (https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python)?
  p = port
  port += 1
  return p

def channel(port, route):
  return "{}/{}".format(port, route)

def rl_node(path, port, node_urls, input_shape, output_shape, name, discrete=False, reward_prop=0, niceness=1, logdir=None):
  #TODO: support multidim in/out shapes later
  node_urls = " ".join(map(str, node_urls))
  #input_shape = " ".join(map(str, input_shape))
  #output_shape = " ".join(map(str, output_shape))
  logdir = "-l {}".format(logdir) if logdir is not None else ""
  command = "python scripts/run_sb.py -u {} -p {} -i {} -o {} -d {} -f {} -s {} -n {} -r {} {}"
  command = command.format(node_urls, port, input_shape, output_shape, discrete, path, niceness, name, reward_prop, logdir)
  subprocess.Popen(command)
  time.sleep(delay)

def env_node(env_name, port, node_urls, proxy=None, name=None, niceness=1):
  node_urls = " ".join(map(str, node_urls))
  name = env_name if name is None else name
  #TODO: hardcoded render
  proxy = ("-x {}".format(proxy)) if proxy is not None else ""
  command = 'python scripts/run_env.py -u {} -p {} -s {} -n {} --render -e "{}" {}'
  command = command.format(node_urls, port, niceness, name, env_name, proxy)
  subprocess.Popen(command)
  time.sleep(delay)

def reward_proxy(port, node_urls, channels=[], niceness=1):
  #TODO: add args for clips/decay rates/scales?
  node_urls = " ".join(map(str, node_urls))
  channels = " ".join(map(str, channels))
  command = 'python scripts/run_reward_proxy.py -p {} -u {} -c {} -s {}'
  command = command.format(port, node_urls, channels, niceness)
  subprocess.Popen(command)
  time.sleep(delay)

#allocate node ports
rl1 = allocate()
rl2 = allocate()
env = allocate()
rewards = allocate()

rp = 0 #reward propagation
nn = 1 #niceness

rl_node("models/sb/double_1", rl1, [env], 8, 32, "rl1", False, rp, logdir="logs/sb", niceness=nn)
rl_node("models/sb/double_2", rl2, [rl1], 32, 4, "rl2", True, rp, logdir="logs/sb", niceness=nn)
env_node("LunarLander-v2", env, [rl2], rewards, niceness=nn)
#env_node("LunarLander-v2", env, [rl2], niceness=nn)
reward_proxy(rewards, [rl1, rl2], niceness=nn)
