from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "mspacman"

Batcher([
  #['python scripts/builder.py -i 128 -o 18 -s 64 -a relu -A softmax -p adam -l 1e-3 -c 1.0 -f models/mspacman/1.h5'],
  ['python scripts/builder.py -i 128 -o 18 -s 64 -a relu -A softmax -p adam -l 1e-4 -c 1.0 -f models/mspacman/1.h5'],
  [
    'python scripts/run_retro.py -u 8201 -p 8200 -s {} -n mspacman -k 4 -a discrete -e "MsPacMan-Atari2600" -o ram --render'.format(niceness),
    'python scripts/run_pget.py -u 8200 -p 8201 -m models/mspacman/1.h5 -t discrete -k 1e-6 -s {} -n {} -d 10 -e 0.01 -g 0.999 -l 0.9'.format(niceness, name)
  ]
]).run()
