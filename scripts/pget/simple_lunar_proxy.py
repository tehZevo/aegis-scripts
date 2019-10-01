from zbatcher import Batcher

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -f models/slp/1.h5'],
  [
    'python scripts/run_reward_proxy.py -u 8001 -p 8002 -s -0.1',
    'python scripts/run_env.py -u 8001 -x 8001 -p 8000 -s -0.1 -n simple_lunar_proxy -e "LunarLander-v2"',
    'python scripts/run_pget.py -u 8000 -p 8001 -m models/slp/1.h5 -d True -s -0.1 -n 0.01 -a 1e-3'
  ]
]).run()
