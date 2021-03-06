from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "simple_lunar"

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -p adam -l 1e-3 -c 999999 -f models/sl/1.h5'],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_pget.py -u 8500 -p 8501 -m models/sl/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -g 0.999 -e 0.1'.format(niceness, name)
  ]
]).run()
