# Aegis Scripts
Scripts for *actually* running [Aegis](https://github.com/tehZevo/aegis-core) nodes.

|Script|Description|
|---|---|
|`batch.py`| Run multiple commands in parallel (see `batch_examples/` for examples)|
|`builder.py`| Allows for quick command-line creation of models compatible with `run_pget.py`|
|`run_env.py`| Starts an OpenAI Gym environment node|
|`run_pget.py`| Starts a [PGET](https://github.com/tehZevo/pget) RL agent node|
|`run_sb_dqn.py`| Starts a [stable-baselines](https://github.com/hill-a/stable-baselines) DQN agent node (WIP)|
|`run_sb_ppo2.py`| Starts a [stable-baselines](https://github.com/hill-a/stable-baselines) PPO2 agent node (WIP)|

## Single-network CartPole example
First, create a suitable network
```
python scripts/builder.py -i 4 -o 2 -s 32 32 -A softmax -f models/cartpole_single.h5
```
Then, launch the environment
```
python scripts/run_env.py -u http://localhost:8001 -p 8000 -r -1
```
Finally, in a separate shell, run a PGET agent with the network we created earlier
```
python scripts/run_pget.py -u http://localhost:8000 -p 8001 -m models/cartpole_single.h5 -d True
```
