"""
Water Filling Algorithm for ZF Beamforming
Author    : Khin Thandar Kyaw
Reference : https://www.scicoding.com/waterfilling/
Date      : 4 Jan 2023
"""

from timer import *
from NNUtils import *
import numpy as np
import tensorflow as tf

totalUsers = totalUsersFunc()

def getGain(W, R):
  W_H = tf.linalg.adjoint(W)
  gain = np.squeeze(tf.math.real(tf.einsum('mqk, mkl, mlp->mqp', W_H, R, W)))
  return gain


for totalUser in totalUsers:
  
  timeArray = np.load(f'test/{totalUser}users/timeArrayZWF.npy')
  timeList = timeArray.tolist()
  with Timer() as timer:
    
    print(f'Calculating for {totalUser} users...')
    covariance = np.load(f'test/{totalUser}users/cov_test.npy')
    _, _, _, _, NoiseVarTotal, _  = dataPreparation(covariance)
    beams = np.load(f'test/{totalUser}users/beamZF.npy')
    
    sumRate = []
    error = []
    for snr in range(-5, 25, 5):
      print(f'Calculating for the sum rate using water-filling algorithm at SNR = {snr} dB...')
      sumRateAll = []
      ErrorAll = []
      for sample in range(covariance.shape[0]):
        R = covariance[sample] # (totalUser, Nt, Nt)
        W = beams[sample] # (totalUser, Nt, 1)
        noise = NoiseVarTotal[sample] #(1,)
        
        SNR = np.power(10, snr / 10)
        
        P_total = SNR * noise # (1,,)
        #print(f'P_total: {P_total}')
        
        gain = getGain(W, R) # (totalUser, 1, 1)
        #print(f'gain: {gain}')
        
        # Bisection search for Lambda
        #lowerBound = 0 # Initial lower bound
        #upperBound = (P_total + np.sum(gain)) # Initial upper bound
        lowerBound = 1 / (P_total + np.sum(1 / gain))
        upperBound = np.max(gain)
        
        tolerance = 1e-7 
        
        while (upperBound - lowerBound) > tolerance:
          mid_point = (lowerBound + upperBound) / 2
          
          # Calculate the power allocation
          p = (1 / mid_point) - (1 / gain)
          p[p < 0] = 0 # consider only positive power allocation
          
          # Test sum-power constraints
          if (np.sum(p) > P_total): # Exceeds power limit => increase the lower bound
            lowerBound = mid_point
          else: # less than the power limit => decrease the upper bound
            upperBound = mid_point
        
        sumRate_m = np.sum(np.log2(1 + np.maximum(0, p * gain)))
        sumRateAll.append(sumRate_m)
        
        error_m = np.abs(P_total - np.sum(p))
        #ErrorAll.append(error_m)
      
      sumRate.append(np.mean(sumRateAll))
      #error.append(np.mean(ErrorAll))
  
  timeList.append(timer.elapsed_time)
  np.save(f'test/{totalUser}users/timeArrayZWF.npy', np.array(timeList))
  
  print(f'sumRate: {sumRate}')
  #print(f'error: {error}')
  ensure_dir(f'Plotting/{totalUser}users/')
  np.save(f'Plotting/{totalUser}users/sumRateWF.npy', np.array(sumRate))

    

    