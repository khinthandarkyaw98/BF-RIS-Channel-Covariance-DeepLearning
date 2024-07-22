import sys
import numpy as np
from nn_utils import *

total_users = total_users()
  
def timer_calculation(time_array):
  time_list = time_array.tolist()
  #print(f'len(timeList): {len(timeList)}')
  time_sum = 0
  for time in time_list:
    time_sum += time
  return time_sum

def print_and_write(file, *args, end='\n'):
  print(*args, end=end)
  file.write(' '.join(map(str, args)) + end)
  
for totalUser in total_users:
  ensure_dir(f'timer/{totalUser}users/')
  with open(f'timer/{totalUser}users/timer.txt', 'w') as file:
    Nt, N, _, _, _, _, _ = parameters(totalUser)
    timeWF = np.load(f'test/{totalUser}users/timeArrayZWF.npy')
    
    print_and_write(file, '=' * 50)
    print_and_write(file, f'Nt = {Nt}, N = {N}, M + K = {totalUser}')
    print_and_write(file, '=' * 50)
    
    print_and_write(file, 'Elapsed time for ZF Beams w/ WF pwr allocation')
    print_and_write(file, f'{timer_calculation(timeWF)} seconds')
    print_and_write(file, '=' * 50)
    
    timeNN = np.load(f'test/{totalUser}users/timeArraySuper.npy')
    print_and_write(file, 'Elapsed time for Unsupervised Neural Network during testing')
    print_and_write(file, f'{timer_calculation(timeNN)} seconds')
    print_and_write(file, '=' * 50)
