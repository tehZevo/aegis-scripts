from zbatcher import Batcher

#niceness = -0.001
niceness = 1
#niceness = -0.01
lr = 1e-4
#action_repeat = 4
action_repeat = 1

Batcher([
  [
    #create models
    'python scripts/builder.py -i 8 -o 32 -s 32 32 -a relu -A tanh -f models/dl/1.h5',
    'python scripts/builder.py -i 8 -o 32 -s 32 32 -a relu -A tanh -f models/dl/2.h5',
    #'python scripts/builder.py -i 32 -o 32 -A tanh -f models/dl/3.h5',
    'python scripts/builder.py -i 32 -o 32 -s 32 -a relu -A tanh -f models/dl/3.h5',
    'python scripts/builder.py -i 32 -o 4 -s 32 32 -a relu -A softmax -f models/dl/4.h5',
    'python scripts/builder.py -i 32 -o 4 -s 32 32 -a relu -A softmax -f models/dl/5.h5'
  ],
  [
    #run reward proxy
    'python scripts/run_reward_proxy.py -p 7999 -c one two -s {} -u 8002 8003 8004 8005 8006'.format(niceness),
    #run envs
    'python scripts/run_env.py -u 8005 -x 7999/one -p 8000 -r -1 -n "dl 1" -e "LunarLander-v2" -s {} -k {}'.format(niceness, action_repeat),
    'python scripts/run_env.py -u 8006 -x 7999/two -p 8001 -r -1 -n "dl 2" -e "LunarLander-v2" -s {} -k {}'.format(niceness, action_repeat),
    #run agents
    'python scripts/run_pget.py -u 8000 -p 8002 -m models/dl/1.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8001 -p 8003 -m models/dl/2.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8002 8003 -p 8004 -m models/dl/3.h5 -s {} -n 0.001 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8004 -p 8005 -m models/dl/4.h5 -e discrete -s {} -n 0.01 -a {}'.format(niceness, lr),
    'python scripts/run_pget.py -u 8004 -p 8006 -m models/dl/5.h5 -e discrete -s {} -n 0.01 -a {}'.format(niceness, lr)
  ]
]).run()
