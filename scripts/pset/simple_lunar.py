from zbatcher import Batcher

#niceness = -0.001
niceness = -0.01
#niceness = 1
name = "pset_simple_lunar"

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -p adam -l 1e-3 -c 1.0 -f models/pset-sl/1.h5'],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_pset.py -u 8500 -p 8501 -m models/pset-sl/1.h5 -t discrete -c 3 -k 1e-5 -s {} -n {} -g 0.999 -e 0.001'.format(niceness, name)
  ]
]).run()
