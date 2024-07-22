""" # Water Filling Algorithm for ZF Beamforming
# Author    : Khin Thandar Kyaw
# Reference : https://www.scicoding.com/waterfilling/
# Date      : 4 Jan 2023


from NNUtils import *
import numpy as np
import tensorflow as tf

#totalUsers = totalUsersFunc()

def getGain(W, R):
  W_H = tf.linalg.adjoint(W)
  gain = np.squeeze(tf.math.real(tf.einsum('mqk, mkl, mlp->mqp', W_H, R, W)))
  return gain

totalUser = 10
#or totalUser in totalUsers:
print(f'Calculating for {totalUser} users...')
covariance = np.load(f'test/{totalUser}users/cov_test.npy')
_, _, _, _, NoiseVarTotal, _  = dataPreparation(covariance)
beams = np.load(f'test/{totalUser}users/beamZF.npy')


#for snr in range(-5, 25, 5):
snr = -5 #dB
print(f'Calculating for the sum rate using water-filling algorithm at SNR = {snr} dB...')
sumRateAll = []
ErrorAll = []

R = covariance[10] # (totalUser, Nt, Nt)
W = beams[10] # (totalUser, Nt, 1)
noise = NoiseVarTotal[10] #(1,)

SNR = np.power(10, snr / 10)

P_total = SNR * noise # (1,,)
#print(f'P_total: {P_total}')

gain = getGain(W, R) # (totalUser, 1, 1)
print(f'gain: {gain}')
print(f'gain.shape: {gain.shape}\n')

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

print()
print(f'Power Allocation: {p}\n')
sumRate_m = np.sum(np.log(1 + (p * gain))/np.log(2))
print(f'sumRateWF: {sumRate_m}\n')

error_m = np.abs(P_total - np.sum(p))
print(f'Power Difference: {error_m}\n')

sumRateZF = np.load(f'Plotting/{totalUser}users/sumRateZF.npy')
print(f'sumRateZF: {sumRateZF[0]}') 






    

 """