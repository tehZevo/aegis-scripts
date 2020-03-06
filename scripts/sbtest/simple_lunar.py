from zbatcher import Batcher

niceness = -0.001
#niceness = 1
name = "sb_simple_lunar"

Batcher([
  [
    'python scripts/build_sb_ppo.py -i 8 -o 4 -d True -p MlpPolicy -f testagent'
  ],
  [
    'python scripts/run_env.py -u 8501 -p 8500 -s {} -n {} --render -e "LunarLander-v2"'.format(niceness, name),
    'python scripts/run_sb.py -u 8500 -p 8501 -i 8 -o 4 -d True -f testagent -s {} -n {}'.format(niceness, name)
  ]
]).run()
