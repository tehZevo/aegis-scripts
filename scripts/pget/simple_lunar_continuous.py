from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "simple_lunar_continuous"

Batcher([
  ['python scripts/builder.py -i 8 -o 2 -s 64 -a relu -A tanh -f models/slc/1.h5'],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "LunarLanderContinuous-v2"'.format(niceness, name),
    'python scripts/run_pget.py -u 8500 -p 8501 -m models/slc/1.h5 -t continuous -c 3 -k 1e-3 -s {} -n {} -g 0.999 -l 0.99 -e 0.001 -a 1e-3'.format(niceness, name)
  ]
]).run()
