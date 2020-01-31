from zbatcher import Batcher

niceness = -0.005
#niceness = 1
name = "montezuma"
repeat = 4
interval = 250

Batcher([
  ['python scripts/builder.py -i 1024 -o 18 -s 64 -a relu -A softmax -p adam -l 1e-4 -c 1.0 -f models/monte/1.h5'],
  #['python scripts/builder.py -i 1024 -o 18 -s 64 -a relu -A softmax -p rmsprop -l 1e-4 -c 1.0 -f models/monte/1.h5'],
  #['python scripts/builder.py -i 1024 -o 18 -s 64 -a relu -A softmax -p sgd -l 1e-3 -c 1.0 -f models/monte/1.h5'],
  ['python scripts/ae_builder.py -i 1024 -l 32 -s 256 -f models/monte/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_retro.py -u 8802 -p 8800 -x 8810 -i {} -k {} -s {} -n {} -o image -a discrete --render -e "MontezumaRevenge-Atari2600"'.format(interval, repeat, niceness, name),
    'python scripts/run_keras_app.py -u 8800 -p 8801 -r 80 80 -s {} -a mobilenet'.format(niceness),
    'python scripts/run_curiosity.py -u 8801 -a 8802 -p 8810 -m models/monte/curiosity.h5 -b True -s {} -n {}'.format(niceness, name),
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -d 100 -g 0.9999 -l 0.99 -e 0.1'.format(niceness, name)
    'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -d 10 -g 0.9999 -l 0.99 -e 0.1'.format(niceness, name)
  ]
]).run()
