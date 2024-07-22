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
from nn_utils import *
from super_utils import *
from keras import layers
import matplotlib.pyplot as plt

# ------------------------------------
# load and generate simulation data
# ------------------------------------
total_users = total_users()
for user_size in total_users:
  print(f'Total # of Users: {user_size}')
  print_line()
  
  antenna_size, _, _, _, _, _, _ = parameters(user_size)
  snr_fixed = fixed_snr()
    
  covariance_complex = np.load(f'train/{user_size}users/cov_train.npy')
  e_max_complex = np.load(f'train/{user_size}users/eMax_train.npy')

  # ----------Prepare Data---------
  batch_size, sample_size, covariance_stacked, snr_total, noise_var_total, power_total  = data_preparation(covariance_complex)

  e_max_stacked = stacking(e_max_complex)

  # (sampleSize, antennaSize, antennaSize)
  identity_matrix = tf.cast(tf.eye(antenna_size, batch_shape=[sample_size]), dtype=tf.complex64)

  # (sampleSize, 1, antennaSize, antennaSize)
  identity_matrix = tf.expand_dims(identity_matrix, axis=1)

  # (sampleSize, userSize, antennaSize, antennaSize)
  identity_matrix = tf.tile(identity_matrix, [1, user_size, 1, 1])


  # ------------------------------------
  # Construct the Unsupervised Model
  # ------------------------------------

  # (userSize, real/imag, anteannaSize, antennaSize)
  covariance_stacked_input = layers.Input(name='CovarianceStackedInput', 
                                        shape=(covariance_stacked.shape[1:5]), 
                                        dtype=tf.float32)
  power_total_input = layers.Input(name='PowerTotalInput', 
                                  shape=(1,), 
                                  dtype=tf.float32)
  identity_matrix_input = layers.Input(name='identityMatrixInput', 
                                      shape=(identity_matrix.shape[1:4]), 
                                      dtype=tf.complex64)
  covariance_complex_input = layers.Input(name='CovarianceComplexInput', 
                                        shape=(covariance_complex.shape[1:4]), 
                                        dtype=tf.complex64)
  e_max_complex_input = layers.Input(name='eMaxComplexInput', 
                                  shape=(e_max_complex.shape[1:4]), 
                                  dtype=tf.complex64)
  e_max_stacked_input = layers.Input(name="eMaxStackedInput", 
                                  shape=(e_max_stacked.shape[1:5]), 
                                  dtype=tf.float32)
  temp1 = layers.BatchNormalization()(covariance_stacked_input)
  temp1 = layers.Flatten()(temp1)
  temp2 = layers.BatchNormalization()(power_total_input)
  temp2 = layers.Flatten()(temp2)
  temp3 = layers.BatchNormalization()(e_max_stacked_input)
  temp3 = layers.Flatten()(temp3)
  temp = layers.concatenate([temp1, temp2, temp3])
  temp = layers.BatchNormalization()(temp)
  # temp = layers.Dense(512, activation='softplus')(temp)
  # temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(256, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(128, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp = layers.Dense(64, activation='softplus')(temp)
  temp = layers.BatchNormalization()(temp)
  temp_first_half = layers.Lambda(lambda x: x[:, :32])(temp)
  temp_second_half = layers.Lambda(lambda x: x[:, 32:])(temp)
  power_temp = layers.Dense(user_size, activation='softplus')(temp_first_half)
  beta_temp = layers.Dense(user_size, activation='softplus')(temp_second_half)
  individual_power_output = layers.Lambda(trans_power, 
                                        dtype=tf.float32, 
                                        output_shape=(user_size, 1, 1))([power_temp, power_total_input])
  individual_beta_output = layers.Lambda(trans_Beta, 
                                        dtype=tf.float32, 
                                        output_shape=(user_size, 1, 1))([beta_temp, power_total_input])
  beam = layers.Lambda(compute_beam, 
                        dtype=tf.complex64, 
                        output_shape=(user_size, antenna_size, 1))([individual_power_output, 
                          individual_beta_output, 
                          e_max_complex_input, 
                          identity_matrix_input, 
                          covariance_complex_input])
  loss = layers.Lambda(loss_func_unsuper, 
                        dtype=tf.float32, 
                        output_shape=(1,))([covariance_complex_input, beam])
  model = keras.Model(inputs=[covariance_stacked_input, 
                                  power_total_input, 
                                  identity_matrix_input, 
                                  covariance_complex_input, 
                                  e_max_complex_input, 
                                  e_max_stacked_input], outputs=loss)
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
  checkpoint = keras.callbacks.ModelCheckpoint(f'train/{user_size}users/trainedSuper.h5', 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=True,
                                                  mode='min', 
                                                  save_weights_only=True)
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    min_delta=0, 
                                                    patience=5, 
                                                    verbose=0, 
                                                    mode='min', 
                                                    restore_best_weights=True)

  history = model.fit(x=[covariance_stacked, power_total, 
                         identity_matrix, covariance_complex, 
                         e_max_complex, e_max_stacked], 
                      y=covariance_complex, # Dummy target
                      batch_size=batch_size,
                      epochs=500,
                      verbose=2,
                      validation_split= 0.3, 
                      callbacks=[reduced_lr, checkpoint, early_stopping]
                      )

  # ------------------------------------
  # Plot the loss curve
  # ------------------------------------
  loss_curve(history, user_size, 'Indiviudal beta and Individual power Constraints', snr_fixed)
  plt.savefig(f'train/{user_size}users/LossCurve_Super.png')
  plt.close()

