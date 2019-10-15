from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "curiosity_lunar"

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -f models/cl/1.h5'],
  #['python scripts/builder.py -i 8 -o 4 -s 64 -a tanh -A softmax -f models/cl/1.h5'],
  #['python scripts/ae_builder.py -i 8 -l 4 -s 64 -f models/cl/curiosity.h5'],
  ['python scripts/ae_builder.py -i 8 -l 16 -s 64 -f models/cl/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_env.py -u 8301 -p 8300 -x 8302 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_curiosity.py -u 8300 -a 8301 -p 8302 -m models/cl/curiosity.h5 -s {} -n {}'.format(niceness, name),
    'python scripts/run_pget.py -u 8300 -p 8301 -m models/cl/1.h5 -t discrete -k 1e-4 -s {} -n {} --alt-trace -g 0.999 -e 0.01 -a 1e-3'.format(niceness, name)
  ]
]).run()
