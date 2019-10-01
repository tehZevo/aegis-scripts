from zbatcher import Batcher

Batcher([
  [
    #create models
    'python scripts/builder.py -i 8 -o 64 -s 64 -A tanh -f models/dl/1.h5',
    'python scripts/builder.py -i 8 -o 64 -s 64 -A tanh -f models/dl/2.h5',
    'python scripts/builder.py -i 64 -o 64 -A tanh -f models/dl/3.h5',
    'python scripts/builder.py -i 64 -o 4 -s 64 -A softmax -f models/dl/4.h5',
    'python scripts/builder.py -i 64 -o 4 -s 64 -A softmax -f models/dl/5.h5'
  ],
  [
    #run reward proxy
    'python scripts/run_reward_proxy.py -p 7999 -s -0.1 -u 8002 8003 8004 8005 8006',
    #run envs
    'python scripts/run_env.py -u 8005 -x 7999 -p 8000 -r -1 -n "dl 1" -e "LunarLander-v2" -s -0.1',
    'python scripts/run_env.py -u 8006 -x 7999 -p 8001 -r -1 -n "dl 2" -e "LunarLander-v2" -s -0.1',
    #run agents
    'python scripts/run_pget.py -u 8000 -p 8002 -m models/dl/1.h5 -s -0.1 -n 0.001 -a 1e-4',
    'python scripts/run_pget.py -u 8001 -p 8003 -m models/dl/2.h5 -s -0.1 -n 0.001 -a 1e-4',
    'python scripts/run_pget.py -u 8002 8003 -p 8004 -m models/dl/3.h5 -s -0.1 -n 0.001 -a 1e-4',
    'python scripts/run_pget.py -u 8004 -p 8005 -m models/dl/4.h5 -d True -s -0.1 -n 0.01 -a 1e-4',
    'python scripts/run_pget.py -u 8004 -p 8006 -m models/dl/5.h5 -d True -s -0.1 -n 0.01 -a 1e-4'
  ]
]).run()
