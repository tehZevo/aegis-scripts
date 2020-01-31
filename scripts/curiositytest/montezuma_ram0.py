from zbatcher import Batcher

niceness = -0.005
#niceness = 1
name = "montezuma-ram"
repeat = 4
interval = 250

Batcher([
  ['python scripts/builder.py -i 128 -o 18 -s 64 -a relu -A softmax -p adam -l 1e-4 -c 99999 -f models/monte-ram/1.h5'],
  ['python scripts/ae_builder.py -i 128 -l 64 -s 64 -f models/monte-ram/curiosity.h5'],
  [
    #until we have --no-reward, just proxy the rewards to the curiosity node, since it doesnt use it
    'python scripts/run_retro.py -u 8801 -p 8800 -x 8810 -i {} -k {} -s {} -n {} -o ram -a discrete --render -e "MontezumaRevenge-Atari2600"'.format(interval, repeat, niceness, name),
    #'python scripts/run_curiosity.py -u 8800 -a 8801 -p 8810 -m models/monte-ram/curiosity.h5 -b True -s {} -n {}'.format(niceness, name),
    'python scripts/run_curiosity.py -u 8800 -a 8801 -p 8810 -m models/monte-ram/curiosity.h5 -b False -s {} -n {}'.format(niceness, name),
    'python scripts/run_pget.py -u 8800 -p 8801 -m models/monte-ram/1.h5 -t discrete -c 3 -k 1e-6 -s {} -n {} -d 10 -g 0.999 -l 0.9 -e 0.1'.format(niceness, name)
  ]
]).run()
