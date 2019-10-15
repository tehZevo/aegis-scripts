from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "curiosity_walker"

Batcher([
  ['python scripts/builder.py -i 24 -o 4 -s 64 -a relu -A tanh -f models/cw/1.h5'],
  #['python scripts/builder.py -i 24 -o 4 -s 64 -a tanh -A tanh -f models/cw/1.h5'],
  #['python scripts/ae_builder.py -i 24 -l 10 -s 64 -f models/cw/curiosity.h5'],
  ['python scripts/ae_builder.py -i 24 -l 32 -s 64 -f models/cw/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_env.py -u 8701 -p 8700 -x 8702 -k 4 -s {} -n {} --render -e "BipedalWalker-v2"'.format(niceness, name),
    'python scripts/run_curiosity.py -u 8700 -a 8701 -p 8702 -m models/cw/curiosity.h5 -s {} -n {}'.format(niceness, name),
    'python scripts/run_pget.py -u 8700 -p 8701 -m models/cw/1.h5 -t continuous -k 1e-3 -s {} -n {} -g 0.999 -e 0.01 -a 1e-3'.format(niceness, name)
  ]
]).run()
