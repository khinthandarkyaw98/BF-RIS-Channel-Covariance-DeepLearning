import sys
import numpy as np
from NNUtils import *

totalUsers = totalUsersFunc()
  
def timerCalculation(timeArray):
  timeList = timeArray.tolist()
  #print(f'len(timeList): {len(timeList)}')
  timeSum = 0
  for time in timeList:
    timeSum += time
  return timeSum

def printAndWrite(file, *args, end='\n'):
  print(*args, end=end)
  file.write(' '.join(map(str, args)) + end)
  
for totalUser in totalUsers:
  ensure_dir(f'timer/{totalUser}users/')
  with open(f'timer/{totalUser}users/timer.txt', 'w') as file:
    Nt, N, _, _, _, _, _ = parameters(totalUser)
    timeWF = np.load(f'test/{totalUser}users/timeArrayZWF.npy')
    
    printAndWrite(file, '=' * 50)
    printAndWrite(file, f'Nt = {Nt}, N = {N}, M + K = {totalUser}')
    printAndWrite(file, '=' * 50)
    
    printAndWrite(file, 'Elapsed time for ZF Beams w/ WF pwr allocation')
    printAndWrite(file, f'{timerCalculation(timeWF)} seconds')
    printAndWrite(file, '=' * 50)
    
    timeNN = np.load(f'test/{totalUser}users/timeArraySuper.npy')
    printAndWrite(file, 'Elapsed time for Unsupervised Neural Network during testing')
    printAndWrite(file, f'{timerCalculation(timeNN)} seconds')
    printAndWrite(file, '=' * 50)
