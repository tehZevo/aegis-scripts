from zbatcher import Batcher

niceness = 1
frameskip = 4
step_limit = 10000

name = "atari_gauntlet"

Batcher([
  [
    'python scripts/build_sb_ppo.py -i 128 -o 18 -d True -p MlpPolicy -f testagent'
  ],
  [
    #'tensorboard --logdir logs',
    #'python scripts/run_atari_gauntlet.py -u 8501 -p 8500 -s {} -n {} -g MsPacMan Pong MontezumaRevenge --render -k {} -l {} -o ram'.format(niceness, name, frameskip, step_limit),
    'python scripts/run_atari_gauntlet.py -u 8501 -p 8500 -s {} -n {} --render -k {} -l {} -o ram'.format(niceness, name, frameskip, step_limit),
    #'python scripts/run_sb.py -u 8500 -p 8501 -i 128 -o 18 -d True -f testagent -l logs/sb -s {} -n {}'.format(niceness, name)
  ]
]).run()
