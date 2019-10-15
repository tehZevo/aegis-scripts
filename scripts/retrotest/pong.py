from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "pong"

Batcher([
  ['python scripts/builder.py -i 128 -o 8 -s 64 -a relu -A sigmoid -f models/pong/1.h5'],
  #['python scripts/builder.py -i 128 -o 8 -s 64 -a tanh -A sigmoid -f models/pong/1.h5'],
  [
    #'python scripts/run_retro.py -u 8401 -p 8400 -s {} -n {} -k 4 -e "Pong-Atari2600" -o ram --render'.format(niceness, name),
    'python scripts/run_retro.py -u 8401 -p 8400 -s {} -n {} -e "Pong-Atari2600" -o ram --render'.format(niceness, name),
    'python scripts/run_pget.py -u 8400 -p 8401 -m models/pong/1.h5 -t multibinary -s {} -n {} -g 0.999 -e 0.1 -a 1e-4'.format(niceness, name)
  ]
]).run()
