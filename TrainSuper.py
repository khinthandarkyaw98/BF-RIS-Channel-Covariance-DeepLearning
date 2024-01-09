"""
Build a unsupervised model for indiviudal power and individual beta optimization
Author  : Khin Thandar Kyaw
Reference : DL Framework for Optimization of MISO Downlink Beamforming, TCOM,
            TianLin0509/BF-design-with-DL
Date    : 8 Nov 2023
Last Modified : 15 Nov 2023
"""

import numpy as np
import tensorflow as tf
import keras
from NNUtils import *
from SuperUtils import *
from keras import layers
import matplotlib.pyplot as plt

# ------------------------------------
# load and generate simulation data
# ------------------------------------
totalUsers = totalUsersFunc()
for userSize in totalUsers:
  print(f'Total # of Users: {userSize}')
  linePrint()
  
  antennaSize, _, _, _, _, _, _ = parameters(userSize)
  snrFixed = fiexdSNR()
    
  covarianceComplex = np.load(f'train/{userSize}users/cov_train.npy')
  eMaxComplex = np.load(f'train/{userSize}users/eMax_train.npy')

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
  optimizer = keras.optimizers.Adam(learning_rate=1e-5)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress warning # cos I believe that the warning is due to internal issues
  model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
  model.summary()

  # ------------------------------------
  # Train the Model
  # ------------------------------------
  reduced_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.1, 
                                                    patience=3, 
                                                    mode= 'min',
                                                    min_delta=0.01,
                                                    min_lr=1e-7)
  checkpoint = keras.callbacks.ModelCheckpoint(f'train/{userSize}users/trainedSuper.h5', 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=True,
                                                  mode='min', 
                                                  save_weights_only=True)
  earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    min_delta=0, 
                                                    patience=5, 
                                                    verbose=0, 
                                                    mode='min', 
                                                    restore_best_weights=True)

  history = model.fit(x=[covarianceStacked, PowerTotal, 
                         identityMatrix, covarianceComplex, 
                         eMaxComplex, eMaxStacked], 
                      y=covarianceComplex, # Dummy target
                      batch_size=batchSize,
                      epochs=500,
                      verbose=2,
                      validation_split= 0.3, 
                      callbacks=[reduced_lr, checkpoint, earlyStopping]
                      )

  # ------------------------------------
  # Plot the loss curve
  # ------------------------------------
  lossCurve(history, userSize, 'Indiviudal beta and Individual power Constraints', snrFixed)
  plt.savefig(f'train/{userSize}users/LossCurve_Super.png')
  plt.close()

