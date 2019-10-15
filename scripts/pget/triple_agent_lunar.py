from zbatcher import Batcher
#seems to work with 1x32 hidden in input/output but 0x hidden in center

#niceness = -0.1
niceness = 1
lr = 1e-4

Batcher([
  [
    #create models
    'python scripts/builder.py -i 8 -o 32 -s 32 -a tanh -A tanh -f models/tal/1.h5',
    'python scripts/builder.py -i 32 -o 32 -a tanh -A tanh -f models/tal/2.h5',
    'python scripts/builder.py -i 32 -o 4 -s 32 -a tanh -A softmax -f models/tal/3.h5'
  ],
  [
    #start proxy
    'python scripts/run_reward_proxy.py -u 8001 8002 8003 -p 7999 -s {}'.format(niceness),
    'python scripts/run_env.py -u 8003 -p 8000 -r -1 -s {} -n tal -e "LunarLander-v2"'.format(niceness),
    'python scripts/run_pget.py -u 8000 -p 8001 -m models/tal/1.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8001 -p 8002 -m models/tal/2.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8002 -p 8003 -m models/tal/3.h5 -e discrete -s {} -n 0.01 -a {}'.format(niceness, lr)
  ]
]).run()
