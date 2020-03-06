from zbatcher import Batcher

niceness = 1
#niceness = 1
name = "sb_double_lunar"

Batcher([
  [
    'python scripts/build_sb_ppo.py -i 8 -o 32 -d False -p MlpPolicy -l 1e-3 -f models/sb/double_1',
    'python scripts/build_sb_ppo.py -i 32 -o 4 -d True -p MlpPolicy -l 1e-3 -f models/sb/double_2',
  ],
  [
    'python scripts/run_env.py -u 8502 -p 8500 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_sb.py -u 8500 -p 8501 -i 8 -o 32 -d False -f models/sb/double_1 -l logs/sb -s {} -n {} -r 1'.format(niceness, name + "_1"),
    'python scripts/run_sb.py -u 8501 -p 8502 -i 32 -o 4 -d True -f models/sb/double_2 -l logs/sb -s {} -n {} -r 1'.format(niceness, name + "_2")
  ]
]).run()
