"""
Water Filling Algorithm for ZF Beamforming
Author    : Khin Thandar Kyaw
Reference : https://www.scicoding.com/waterfilling/
Date      : 4 Jan 2023
"""

from timer import *
from nn_utils import *
import numpy as np
import tensorflow as tf

total_users = total_users()

def get_gain(W, R):
  W_H = tf.linalg.adjoint(W)
  gain = np.squeeze(tf.math.real(tf.einsum('mqk, mkl, mlp->mqp', W_H, R, W)))
  return gain


for totalUser in total_users:
  
  time_array = np.load(f'test/{totalUser}users/timeArrayZWF.npy')
  time_list = time_array.tolist()
  with Timer() as timer:
    
    print(f'Calculating for {totalUser} users...')
    covariance = np.load(f'test/{totalUser}users/cov_test.npy')
    _, _, _, _, noise_var_total, _  = data_preparation(covariance)
    beams = np.load(f'test/{totalUser}users/beamZF.npy')
    
    sum_rate = []
    error = []
    for snr in range(-5, 25, 5):
      print(f'Calculating for the sum rate using water-filling algorithm at SNR = {snr} dB...')
      sum_rate_all = []
      error_all = []
      for sample in range(covariance.shape[0]):
        R = covariance[sample] # (totalUser, Nt, Nt)
        W = beams[sample] # (totalUser, Nt, 1)
        noise = noise_var_total[sample] #(1,)
        
        SNR = np.power(10, snr / 10)
        
        P_total = SNR * noise # (1,,)
        #print(f'P_total: {P_total}')
        
        gain = get_gain(W, R) # (totalUser, 1, 1)
        #print(f'gain: {gain}')
        
        # Bisection search for Lambda
        #lowerBound = 0 # Initial lower bound
        #upperBound = (P_total + np.sum(gain)) # Initial upper bound
        lower_bound = 1 / (P_total + np.sum(1 / gain))
        upper_bound = np.max(gain)
        
        tolerance = 1e-7 
        
        while (upper_bound - lower_bound) > tolerance:
          mid_point = (lower_bound + upper_bound) / 2
          
          # Calculate the power allocation
          p = (1 / mid_point) - (1 / gain)
          p[p < 0] = 0 # consider only positive power allocation
          
          # Test sum-power constraints
          if (np.sum(p) > P_total): # Exceeds power limit => increase the lower bound
            lower_bound = mid_point
          else: # less than the power limit => decrease the upper bound
            upper_bound = mid_point
        
        sum_rate_m = np.sum(np.log2(1 + np.maximum(0, p * gain)))
        sum_rate_all.append(sum_rate_m)
        
        error_m = np.abs(P_total - np.sum(p))
        #ErrorAll.append(error_m)
      
      sum_rate.append(np.mean(sum_rate_all))
      #error.append(np.mean(ErrorAll))
  
  time_list.append(timer.elapsed_time)
  np.save(f'test/{totalUser}users/timeArrayZWF.npy', np.array(time_list))
  
  print(f'sumRate: {sum_rate}')
  #print(f'error: {error}')
  ensure_dir(f'Plotting/{totalUser}users/')
  np.save(f'Plotting/{totalUser}users/sumRateWF.npy', np.array(sum_rate))

    

    