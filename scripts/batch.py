import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--delay", required=False, type=float, default=1)
parser.add_argument("file", type=str)

args = parser.parse_args()

import subprocess
import time

with open(args.file, 'r') as file:
    commands = file.read().splitlines()

delay = args.delay

processes = []

def wait_for_all():
  while len(processes) > 0:
    #if first process is closed
    if processes[0].poll() is not None:
      #remove
      processes.pop(0)
    #otherwise wait a bit
    else:
      time.sleep(delay)

for command in commands:
  #wait for all processes to complete on newline
  if command.strip() == "":
    wait_for_all()
  #skip lines beginning with hash
  elif command[0] == "#":
    continue
  #otherwise run command
  else:
    processes.append(subprocess.Popen(command))
    time.sleep(delay)

wait_for_all()
