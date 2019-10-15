import retro
from zbatcher import Batcher
import random

games = retro.data.list_games()
games = filter(lambda x: "atari2600" in x.lower(), games)
game = random.choice(games)

#TODO:
niceness = -0.001
frameskip = 4
#niceness = 1
initial_d = 100

start_port = 9300
port =

center_size = 64
center_port =

def add_pget()

def add_game(game):
  builder =

Batcher([
  ['python scripts/builder.py -i 128 -o 8 -s 64 64 -a relu -A sigmoid -f models/mspacman/1.h5'],
  [
    'python scripts/run_retro.py -u 8101 -p 8100 -s {} -n mspacman -k 4 -e {} -o ram --render'.format(niceness, game),
    'python scripts/run_pget.py -u 8100 -p 8101 -m models/mspacman/1.h5 -e multibinary -s {} -n 0.1 -a 1e-3'.format(niceness)
  ]
]).run()
