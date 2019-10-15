from zbatcher import Batcher

#niceness = -0.001
niceness = 1


Batcher([
  ['python scripts/builder.py -i 3 -o 1 -s 64 64 -a relu -A tanh -f models/sp/1.h5'],
  [
    'python scripts/run_env.py -u 8101 -p 8100 -s {} -n simple_pendulum -e "Pendulum-v0"'.format(niceness),
    'python scripts/run_pget.py -u 8100 -p 8101 -m models/sp/1.h5 -s {} -n 0.1 -a 1e-3'.format(niceness)
  ]
]).run()
