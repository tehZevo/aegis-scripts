from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "curiosity_lunar"

Batcher([
  ['python scripts/builder.py -i 8 -o 4 -s 64 -a relu -A softmax -p adam -l 1e-4 -c 1.0 -f models/cl/1.h5'],
  #['python scripts/ae_builder.py -i 8 -l 4 -s 64 -f models/cl/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_env.py -u 8301 -p 8300 -x 8302 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_curiosity.py -u 8300 -a 8301 -p 8302 -m models/cl/curiosity.h5 -b True -s {} -n {}'.format(niceness, name),
    'python scripts/run_pget.py -u 8300 -p 8301 -m models/cl/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -g 0.999 -e 0.1'.format(niceness, name)
  ]
]).run()
