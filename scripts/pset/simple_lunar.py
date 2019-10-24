from zbatcher import Batcher

#niceness = -0.001
niceness = -0.01
#niceness = 1
name = "pset_simple_lunar"

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -f models/pset-sl/1.h5'],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    #'python scripts/run_pset.py -u 8500 -p 8501 -m models/pset-sl/1.h5 -t discrete -k 1e-1 -s {} -n {} -g 0.999 -e 0.001 -a 1e-3'.format(niceness, name)
    #'python scripts/run_pset.py -u 8500 -p 8501 -m models/pset-sl/1.h5 -t discrete -k 1e-2 -c 3 -s {} -n {} -g 0.999 -e 1e-1 -a 1e-3'.format(niceness, name)
    'python scripts/run_pset.py -u 8500 -p 8501 -m models/pset-sl/1.h5 -t discrete -c 999999 -s {} -n {} -g 0.999 -l 0.99 -e 1e-2 -a 1e-2'.format(niceness, name)
  ]
]).run()
