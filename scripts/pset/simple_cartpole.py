from zbatcher import Batcher

#niceness = -0.001
#niceness = -0.01
niceness = -1/60
#niceness = 1
name = "pset_simple_cartpole"

Batcher([
  ['python scripts/builder.py -i 4 -o 2 -s 64 -a relu -A softmax -f models/pset-sc/1.h5'],
  [
    'python scripts/run_env.py -u 8601 -p 8600 -s {} -n {} --render -r -1 -e "CartPole-v0"'.format(niceness, name),
    #'python scripts/run_pset.py -u 8600 -p 8601 -m models/pset-sc/1.h5 -t discrete -k 1e-1 -s {} -n {} -g 0.999 -e 0.001 -a 1e-3'.format(niceness, name)
    #'python scripts/run_pset.py -u 8600 -p 8601 -m models/pset-sc/1.h5 -t discrete -k 1e-5 -c 9999999 -s {} -n {} -g 0.999 -e 0.01 -a 1e-3'.format(niceness, name)
    #'python scripts/run_pset.py -u 8600 -p 8601 -m models/pset-sc/1.h5 -t discrete -k 1e-3 -c 999999 -s {} -n {} -g 0.999 -l 0.99 -e 1e-2 -a 1e-2'.format(niceness, name)
    #'python scripts/run_pset.py -u 8600 -p 8601 -m models/pset-sc/1.h5 -t discrete -k 1e-2 -c 999999 -s {} -n {} -g 0.999 -l 0.99 -e 1e-2 -a 1e-3'.format(niceness, name)
    'python scripts/run_pset.py -u 8600 -p 8601 -m models/pset-sc/1.h5 -o none -t discrete -k 1e-5 -c 999999 -s {} -n {} -g 0.999 -l 0.99 -e 1e-1 -a 1e-1'.format(niceness, name)
  ]
]).run()
