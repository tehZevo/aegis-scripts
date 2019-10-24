from zbatcher import Batcher

niceness = -0.005
#niceness = 1
name = "montezuma"
repeat = 4
interval = 250

#linear output of AE is probably causing HUGE losses at the beginning...

Batcher([
  #['python scripts/builder.py -i 1024 -o 8 -s 64 -a tanh -A sigmoid -f models/monte/1.h5'],
  ['python scripts/builder.py -i 1024 -o 18 -s 64 -a relu -A softmax -f models/monte/1.h5'],
  #['python scripts/builder.py -i 1024 -o 18 -s 64 -a tanh -A softmax -f models/monte/1.h5'],
  #['python scripts/ae_builder.py -i 1024 -l 32 -s 256 -f models/monte/curiosity.h5'],
  ['python scripts/ae_builder.py -i 1024 -l 256 -f models/monte/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_retro.py -u 8802 -p 8800 -x 8810 -i {} -k {} -s {} -n {} -a discrete --render -e "MontezumaRevenge-Atari2600"'.format(interval, repeat, niceness, name),
    'python scripts/run_keras_app.py -u 8800 -p 8801 -r 80 80 -s {} -a mobilenet'.format(niceness),
    'python scripts/run_curiosity.py -u 8801 -a 8802 -p 8810 -m models/monte/curiosity.h5 -s {} -n {}'.format(niceness, name),
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t multibinary -c 3 -k 1e-5 -s {} -n {} -o none --alt-trace -g 0.999 -e 0.01 -a 1e-3'.format(niceness, name)
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t multibinary -o none -c 2 -k 1e-5 -s {} -n {} -d 1 -g 0.9999 -e 0.1 -a 1e-4'.format(niceness, name)
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t multibinary -o none -c 2 -k 1e-5 -s {} -n {} -d 1 -g 0.9999 -e 0.1 -a 1e-4'.format(niceness, name)
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t discrete -c 2 -k 1e-5 -s {} -n {} -d 1 -g 0.9999 -e 0.1 -a 1e-4'.format(niceness, name)
    #'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -t discrete -c 2 -k 1e-5 -s {} -n {} -d 10 -g 0.9999 -e 0.1 -a 1e-2'.format(niceness, name)
    'python scripts/run_pget.py -u 8801 -p 8802 -m models/monte/1.h5 -o none -t discrete -c 2 -k 1e-5 -s {} -n {} -d 10 -g 0.9999 -e 0.1 -a 1e-3'.format(niceness, name)
  ]
]).run()
