from zbatcher import Batcher

niceness = -0.001
#niceness = 1

Batcher([
  ['python scripts/builder.py -i 128 -o 8 -s 64 -a relu -A sigmoid -f models/mspacman/1.h5'],
  #['python scripts/builder.py -i 128 -o 8 -s 64 -a tanh -A sigmoid -f models/mspacman/1.h5'],
  [
    'python scripts/run_retro.py -u 8201 -p 8200 -s {} -n mspacman -k 4 -e "MsPacMan-Atari2600" -o ram --render'.format(niceness),
    #'python scripts/run_pget.py -u 8200 -p 8201 -m models/mspacman/1.h5 -c 10000 -d 100 -e multibinary -s {} -n 0.1 -l 0.99 -a 1e-3'.format(niceness)
    #'python scripts/run_pget.py -u 8200 -p 8201 -m models/mspacman/1.h5 -e multibinary -s {} -n 0.1 -l 0.99 -a 1e-3'.format(niceness)
    #'python scripts/run_pget.py -u 8200 -p 8201 -m models/mspacman/1.h5 -e multibinary -s {} -n 0.1 -g 0.999 -l 0.9 -a 1e-3'.format(niceness)
    'python scripts/run_pget.py -u 8200 -p 8201 -m models/mspacman/1.h5 -e multibinary -o none -s {} -n 0.1 -g 0.999 -l 0.99 -a 1e-2'.format(niceness)
  ]
]).run()
