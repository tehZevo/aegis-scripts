from zbatcher import Batcher

niceness = -0.001
#niceness = 1

Batcher([
  ['python scripts/builder.py -i 1024 -o 8 -s 64 -a relu -A sigmoid -f models/mn_mpm/1.h5'],
  [
    'python scripts/run_retro.py -u 8102 -p 8100 -s {} -n mn_mpm -k 4 -e "MsPacMan-Atari2600" --render'.format(niceness),
    'python scripts/run_keras_app.py -u 8100 -p 8101 -r 80 80 -s {} -a mobilenet'.format(niceness),
    #'python scripts/run_pget.py -u 8101 -p 8102 -m models/mn_mpm/1.h5 -c 10000 -d 100 -e multibinary -s {} -n 0.1 -l 0.99 -a 1e-3'.format(niceness)
    #'python scripts/run_pget.py -u 8101 -p 8102 -m models/mn_mpm/1.h5 -e multibinary -s {} -n 0.1 -l 0.99 -a 1e-3'.format(niceness)
    #'python scripts/run_pget.py -u 8101 -p 8102 -m models/mn_mpm/1.h5 -e multibinary -s {} -n 0.1 -g 0.999 -l 0.9 -a 1e-3'.format(niceness)
    'python scripts/run_pget.py -u 8101 -p 8102 -m models/mn_mpm/1.h5 -o none -e multibinary -s {} -n 0.1 -d 100 -g 0.999 -l 0.9 -a 1e-1'.format(niceness)
  ]
]).run()
