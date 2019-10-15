from zbatcher import Batcher

niceness = -0.001
#niceness = 1

Batcher([
  ['python scripts/builder.py -i 128 -o 8 -s 64 -a relu -A sigmoid -f models/mspacman/1.h5'],
  [
    'python scripts/run_retro.py -u 8101 -p 8100 -s {} -n mspacman -k 4 -e "MsPacMan-Atari2600" -o ram --render'.format(niceness),
    'python scripts/run_pget.py -u 8100 -p 8101 -m models/mspacman/1.h5 -o none -c 10000 -d 100 -e multibinary -s {} -n 0.1 -l 0.99 -a 1e-1'.format(niceness)
  ]
]).run()
