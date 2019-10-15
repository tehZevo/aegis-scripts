from zbatcher import Batcher

#niceness = -0.1
niceness = 1
lr = 1e-4

Batcher([
  [
    #create models
    'python scripts/builder.py -i 8 -o 32 -s 32 -a tanh -A tanh -f models/tall/1.h5',
    'python scripts/builder.py -i 32 -o 32 -a tanh -A tanh -f models/tall/2.h5',
    'python scripts/builder.py -i 32 -o 4 -s 32 -a tanh -A softmax -f models/tall/3.h5'
  ],
  [
    #start proxy
    'python scripts/run_reward_proxy.py -u 8001 8002 8003 -p 7999 -s {}'.format(niceness),
    'python scripts/run_env.py -u 8003 -p 8000 -r -1 -s {} -n tall -e "LunarLander-v2"'.format(niceness),
    'python scripts/run_pget.py -u 8000 -p 8001 -m models/tall/1.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8001 8002 -p 8002 -m models/tall/2.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8002 -p 8003 -m models/tall/3.h5 -e discrete -s {} -n 0.01 -a {}'.format(niceness, lr)
  ]
]).run()
