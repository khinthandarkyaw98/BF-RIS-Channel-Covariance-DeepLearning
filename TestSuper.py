"""
Build a unsupervised model for indiviudal power and individual beta optimization
Author  : Khin Thandar Kyaw
Reference : DL Framework for Optimization of MISO Downlink Beamforming, TCOM, March 2020
Date    : 8 Nov 2023
Last Modified : 15 Nov 2023
"""

import numpy as np
import tensorflow as tf
import keras
from NNUtils import *
from SuperUtils import *
from keras import layers

# ------------------------------------
# load and generate simulation data
# ------------------------------------
totalUsers = totalUsersFunc()
for userSize in totalUsers:
  print(f'Total # of Users: {userSize}')
  linePrint()
  
  antennaSize, _, _, _, _, _, _ = parameters(userSize)
  snrFixed = fiexdSNR()
  
  covarianceComplex = np.load(f'test/{userSize}users/cov_test.npy')
  eMaxComplex = np.load(f'test/{userSize}users/eMax_test.npy')

  # ----------Prepare Data---------
  batchSize, sampleSize, covarianceStacked, SNRTotal, NoiseVarTotal, PowerTotal  = dataPreparation(covarianceComplex)

  eMaxStacked = stacking(eMaxComplex)

  # (sampleSize, antennaSize, antennaSize)
  identityMatrix = tf.cast(tf.eye(antennaSize, batch_shape=[sampleSize]), dtype=tf.complex64)

  # (sampleSize, 1, antennaSize, antennaSize)
  identityMatrix = tf.expand_dims(identityMatrix, axis=1)

  # (sampleSize, userSize, antennaSize, antennaSize)
  identityMatrix = tf.tile(identityMatrix, [1, userSize, 1, 1])

  # ------------------------------------
  # Construct the Supervised Model
  # ------------------------------------

  # (userSize, real/imag, anteannaSize, antennaSize)
  covarianceStackedInput = layers.Input(name='CovarianceStackedInput', 
                                        shape=(covarianceStacked.shape[1:5]), 
                                        dtype=tf.float32)
  powerTotalInput = layers.Input(name='PowerTotalInput', 
                                  shape=(1,), 
                                  dtype=tf.float32)
  identityMatrixInput = layers.Input(name='identityMatrixInput', 
                                      shape=(identityMatrix.shape[1:4]), 
                                      dtype=tf.complex64)
  covarianceComplexInput = layers.Input(name='CovarianceComplexInput', 
                                        shape=(covarianceComplex.shape[1:4]), 
                                        dtype=tf.complex64)
  eMaxComplexInput = layers.Input(name='eMaxComplexInput', 
                                  shape=(eMaxComplex.shape[1:4]), 
                                  dtype=tf.complex64)
  eMaxStackedInput = layers.Input(name="eMaxStackedInput", 
                                  shape=(eMaxStacked.shape[1:5]), 
                                  dtype=tf.float32)
  temp1 = layers.BatchNormalization()(covarianceStackedInput)
  temp1 = layers.Flatten()(temp1)
  temp2 = layers.BatchNormalization()(powerTotalInput)
  temp2 = layers.Flatten()(temp2)
  temp3 = layers.BatchNormalization()(eMaxStackedInput)
  temp3 = layers.Flatten()(temp3)
  temp = layers.concatenate([temp1, temp2, temp3])
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(512, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(256, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(128, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(64, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  tempFirstHalf = layers.Lambda(lambda x: x[:, :32])(temp)
  tempSecondHalf = layers.Lambda(lambda x: x[:, 32:])(temp)
  powerTemp = layers.Dense(userSize, activation='softplus')(tempFirstHalf)
  betaTemp = layers.Dense(userSize, activation='softplus')(tempSecondHalf)
  individualPowerOutput = layers.Lambda(transPower, 
                                        dtype=tf.float32, 
                                        output_shape=(userSize, 1, 1))([powerTemp, powerTotalInput])
  individualBetaOutput = layers.Lambda(transBeta, 
                                        dtype=tf.float32, 
                                        output_shape=(userSize, 1, 1))([betaTemp, powerTotalInput])
  beam = layers.Lambda(computeBeam, 
                        dtype=tf.complex64, 
                        output_shape=(userSize, antennaSize, 1))([individualPowerOutput, 
                          individualBetaOutput, 
                          eMaxComplexInput, 
                          identityMatrixInput, 
                          covarianceComplexInput])
  loss = layers.Lambda(lossFuncSuper, 
                        dtype=tf.float32, 
                        output_shape=(1,))([covarianceComplexInput, beam])
  model = keras.Model(inputs=[covarianceStackedInput, 
                                  powerTotalInput, 
                                  identityMatrixInput, 
                                  covarianceComplexInput, 
                                  eMaxComplexInput, 
                                  eMaxStackedInput], outputs=loss)
  optimizer = keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
  model.summary()
  
  keras.utils.plot_model(model, to_file=f'test/{userSize}users/superModel.png', show_shapes=True, show_layer_names=True, dpi=300)

  # ------------------------------------
  # Save optimal beta and Power values
  # ------------------------------------
  class SavePowerBeta(keras.callbacks.Callback):
    def __init__(self, save_path1, save_path2):
      super(SavePowerBeta, self).__init__()
      self.save_path1 = save_path1
      self.save_path2 = save_path2
      
    def on_test_end(self, logs=None):
      powerBetaModel = keras.Model(inputs=[covarianceStackedInput, 
                                           powerTotalInput, eMaxStackedInput], 
                                      outputs=[individualPowerOutput, individualBetaOutput])
      powerk, betak = powerBetaModel.predict([covarianceStacked, PowerTotal, eMaxStacked])
      np.save(self.save_path1, powerk)
      np.save(self.save_path2, betak)

  # ------------------------------------
  # Train the Model
  # ------------------------------------
  model.load_weights(f'train/{userSize}users/trainedSuper.h5')
  saveOnEval = SavePowerBeta(f'test/{userSize}users/powerk.npy', 
                             f'test/{userSize}users/betak.npy')
  model.evaluate(x=[covarianceStacked, PowerTotal,
                    identityMatrix, covarianceComplex,
                    eMaxComplex, eMaxStacked], 
                 y=covarianceComplex, # Dummy target
                batch_size=batchSize, verbose=0, 
                callbacks=saveOnEval
                )

  # ------------------------------------
  # Compute the sum rate
  # ------------------------------------
  pk = np.load(f'test/{userSize}users/powerk.npy')
  bk = np.load(f'test/{userSize}users/betak.npy')

  print(f'pk.shape: {pk.shape}')
  print(f'bk.shape: {bk.shape}')

  W = computeBeam([pk, bk, eMaxComplex, identityMatrix, covarianceComplex])

  ensure_dir(f'Plotting/{userSize}users/')
  np.save(f'Plotting/{userSize}users/beamFromBetaSuper.npy', W)

  #print(f'W.shape: {W.shape}')
  beamNormSquared = computeNormSquared(W)[0:5]
  print(f'beamNormSquared of Beta NN Model without normalizing: {beamNormSquared}')

  # Check the sum of the norm squared = Power
  print(f'PowerTotal[0: 5]= {PowerTotal[0: 5]}')

  WZ = np.load(f'test/{userSize}users/beamZF.npy')
  print(f'WZ.shape: {WZ.shape}')

  #  Check norm of each beam for ZF = 1
  print('Check if the norm of each beam of ZF = 1')
  beamZNorm = computeNorm(WZ)[0:5]
  print(f'beamZNormSquared: {beamZNorm}')

  print('Loading...')

  rate = []
  for snr in range(-5, 25, 5):
    SNR = np.power(10, np.ones([sampleSize, 1]) * snr / 10)
    Power = SNR * NoiseVarTotal
    normalizedpk = normalization(pk, Power)
    normalizedbk = normalization(bk, Power)
    W = computeBeam([normalizedpk, normalizedbk, 
                     eMaxComplex, identityMatrix, covarianceComplex])
    sumRate = np.mean(computeSumRate(W, covarianceComplex))
    rate.append(sumRate)
    
  np.save(f'Plotting/{userSize}users/sumRateSuper.npy', np.array(rate))
  print("Done!")

