#!/home/jack/anaconda3/bin/python
import os 
import sys
import signal

is_sigint_up = False

def sigint_handler(signum, frame):
  global is_sigint_up
  is_sigint_up = True
  print ('Interrupted!')

signal.signal(signal.SIGINT, sigint_handler)


if len(sys.argv) < 2:
    print("Ooops. Usage:" + sys.argv[0] + ' times')
    exit()

cmd = 'python solve.py'

times = int(sys.argv[1])

for i in range(times):
    if is_sigint_up == True:
        break
    os.system(cmd)
