"""
Generate covariance 
Author    : Khin Thandar Kyaw
Date      : 31 AUG 2023
Last Modified  : 12 JAN 2024
"""

import time
from NNUtils import *
from covarianceUtils import *
from timer import *
  
########################################################################
print('Generating covariance matrix...')
print('Loading...')


totalUsers = totalUsersFunc()

for totalUser in totalUsers:
  covarianceSample = []
  eMaxSample = []
  beamSample = []
  absZFBFSample = []
  sampleSize = 50000
  trainSize = int(0.85 * sampleSize)
  timeList = []
  print(f'Total # of Users: {totalUser}')
  linePrint()
  for sample in range(sampleSize):
    if sample % 1000 == 0:
      print(f'Generating {sample}th sample...')
      linePrint()
    # --------------------------------------------
    # new parameters for M and K for each sample
    # -------------------------------------------
    Nt, N, M, K, Lm, Lk, Ltotal = parameters(totalUser)
    covarianceClass = Covariance(Nt, N, totalUser, M, K, Lm, Lk, Ltotal)
    
    # ------------------------------------
    # direct user channel
    # ------------------------------------
    theta = covarianceClass.generate_theta()
    steering_vectors = covarianceClass.generate_steering_vectors(theta)
    channel_covariance = covarianceClass.generate_channel_covariance(steering_vectors)
    
    # ------------------------------------
    # IRS-assisted user channel
    # ------------------------------------
    xi = covarianceClass.generate_xi()
    upsilon = covarianceClass.generate_upsilon()
    channelBsIrs = covarianceClass.generate_channelBSIRS(xi, upsilon)
    big_theta = covarianceClass.generate_big_theta()
    phi = covarianceClass.generate_phi()
    steering_vectors_irs = covarianceClass.generate_steering_vectors_irs(phi)
    #print(f'Rank of channel_covariance: {np.linalg.matrix_rank(channel_covariance)}')
    channel_covariance_irs = covarianceClass.generate_channel_covariance_irs(steering_vectors_irs)
    #print(f'Rank of channel_covariance_irs: {np.linalg.matrix_rank(channel_covariance_irs)}')
    channel_covariance_all = covarianceClass.generate_composite_channel_covariance(channelBsIrs, big_theta, channel_covariance_irs, channel_covariance)
    #print(f'Rank of channel_covariance_irs_compostite: {np.linalg.matrix_rank(channel_covariance_irs_compostite)}')
    covarianceSample.append(channel_covariance_all)
    
    # save eigenvectors corresponding to the largest eigenvalues
    eMax = covarianceClass.eMax(channel_covariance_all)
    eMaxSample.append(eMax)
    
    # save for zero-forcing
    # ------------------------------------
    # Count the time for ZFBF
    # ------------------------------------
    if sample >= trainSize:
      with Timer() as timer:
        U_tilde, W = performCalculations(covarianceClass, channel_covariance_all)
        beamSample.append(W)
      timeList.append(timer.elapsed_time)
    # ------------------------------------
    else: 
      U_tilde, W = performCalculations(covarianceClass, channel_covariance_all)
      beamSample.append(W)
      
    abs_ZFBF_res = covarianceClass.check_ZFBF_condition(U_tilde, W)
    absZFBFSample.append(abs_ZFBF_res)
      
  covarianceSample= np.array(covarianceSample)
  print(f'channel_covariance.shape: {covarianceSample.shape}')

  eMaxSample = np.array(eMaxSample)
  print(f'eMaxSample.shape: {eMaxSample.shape}')

  beamSample = np.array(beamSample)
  print(f'beamSample.shape: {beamSample.shape}')

  absZFBFSample = np.array(absZFBFSample)
  #print(f'absZFBFSample.shape: {absZFBFSample.shape}')
  #print(f'absZFBFSample[0]: {absZFBFSample[0]}')

  # ------------------------------------
  # split the data
  # -----------------------------------
  # idx is an array of the sample_size
  idx = np.arange(covarianceSample.shape[0])
  # shuffle the idx
  np.random.shuffle(idx)

  #trainSize = int( covarianceSample.shape[0])

  # save channel covariance

  cov_train = covarianceSample[idx[:trainSize]]
  cov_test = covarianceSample[idx[trainSize:]]

  print(f'cov_train.shape: {cov_train.shape}')
  print(f'cov_test.shape: {cov_test.shape}')

  # save eigenVectors corresponding to the largest eigenvalues
  eMax_train = eMaxSample[idx[:trainSize]]
  eMax_test = eMaxSample[idx[trainSize:]]

  print(f'eMax_train.shape: {eMax_train.shape}')
  print(f'eMax_test.shape: {eMax_test.shape}')

  # save beamforming vector
  beam_test = beamSample[idx[trainSize:]]

  print(f'beam_test.shape: {beam_test.shape}')

  print('Saving...')
          
  # Before saving, ensure the dircetory exists
  ensure_dir(f'train/{totalUser}users/')
  ensure_dir(f'test/{totalUser}users/')
  # Now, save the data.
  np.save(f'train/{totalUser}users/cov_train.npy', cov_train) 
  np.save(f'test/{totalUser}users/cov_test.npy', cov_test)
  np.save(f'train/{totalUser}users/eMax_train.npy', eMax_train)
  np.save(f'test/{totalUser}users/eMax_test.npy', eMax_test)
  np.save(f'test/{totalUser}users/beamZF.npy', beam_test) # ZF beamforming
  np.save(f'test/{totalUser}users/timeArrayZWF.npy', timeList) # W Time
  np.save(f'train/{totalUser}users/absZFBFSample.npy', absZFBFSample) # ZF condition
  print(f'Data saved successfully!')
  linePrint()
  
  # ------------------------------------
  # Calculate the sum rate of Zero-Forcing
  # ------------------------------------
  # print("Calculating the sum rate of Zero-Forcing...")
  # print("Loading...")
  
  # rateZ = []
  # _, _, _, _, NoiseVarTotal, _  = dataPreparation(cov_test)
  # for snr in range(-5, 25, 5):
  #   SNR = np.power(10, np.ones([cov_test.shape[0], 1]) * snr / 10)
  #   Power = SNR * NoiseVarTotal
    
  #   # sum rate formulat for wZF is different in noise part
  #   # K / P_total
  #   scaledFactor = np.squeeze(np.sqrt(Power/ totalUser))
  #   sumRateZ = np.mean(computeSumRate(beam_test, cov_test, scaledFactor))
  #   rateZ.append(sumRateZ)
    
  # ensure_dir(f'Plotting/{totalUser}users/')
  # np.save(f'Plotting/{totalUser}users/sumRateZF.npy', np.array(rateZ))
  #print(f'Saved sumRateZ successfully for {totalUser} users!')
  #linePrint()
  

    

  



