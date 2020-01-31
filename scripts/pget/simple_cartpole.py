from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "simple_cartpole"

Batcher([
  ['python scripts/builder.py -i 4 -o 2 -s 64 -a relu -A softmax -p adam -l 1e-3 -c 1.0 -f models/sc/1.h5'],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "CartPole-v0"'.format(niceness, name),
    'python scripts/run_pget.py -u 8500 -p 8501 -m models/sc/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -g 0.999 -e 0.01'.format(niceness, name)
  ]
]).run()
