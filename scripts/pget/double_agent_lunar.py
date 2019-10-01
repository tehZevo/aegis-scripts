from zbatcher import Batcher
#got it to work, needed lower noise on the continuous outputs

Batcher([
  [
    #create models
    'python scripts/builder.py -i 8 -o 32 -s 32 32 -a tanh -A tanh -f models/dal/1.h5',
    'python scripts/builder.py -i 32 -o 4 -s 32 32 -a tanh -A softmax -f models/dal/2.h5'
  ],
  [
    #start proxy
    'python scripts/run_reward_proxy.py -u 8001 8002 -p 7999 -s -0.1',
    #start env
    'python scripts/run_env.py -u 8002 -x 7999 -p 8000 -r -1 -s -0.1 -n dal -e "LunarLander-v2"',
    #start agents
    'python scripts/run_pget.py -u 8000 -p 8001 -m models/dal/1.h5 -s -0.1 -n 0.001 -a 1e-4',
    'python scripts/run_pget.py -u 8001 -p 8002 -m models/dal/2.h5 -d True -s -0.1 -n 0.01 -a 1e-4'
  ]
]).run()
