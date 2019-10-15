from zbatcher import Batcher

niceness = -0.01
#niceness = 1
repeat = 1

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -f models/sl/1.h5'],
  [
    'python scripts/run_env.py -u 8101 -p 8100 -s {} -k {} -n simple_lunar -e "LunarLander-v2"'.format(niceness, repeat),
    'python scripts/run_pget.py -u 8100 -p 8101 -m models/sl/1.h5 -e discrete -s {} -n 0.01 -a 1e-3'.format(niceness)
  ]
]).run()
