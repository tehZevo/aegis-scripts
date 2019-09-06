# Aegis Scripts
Scripts for actually running nodes in [Aegis Core](https://github.com/tehZevo/aegis-core).

## Single-network CartPole example
First, create a suitable network
```
python scripts/builder.py -i 4 -o 2 -s 32 32 -A softmax -f models/cartpole_single.h5
```
Then, launch the environment
```
python scripts/run_env.py -u http://localhost:8001 -p 8000
```
Finally, in a separate shell, run a PGET agent with the network we created earlier
```
python scripts/run_pget.py -u http://localhost:8000 -p 8001 -m models/cartpole_single.h5 -d True
```
