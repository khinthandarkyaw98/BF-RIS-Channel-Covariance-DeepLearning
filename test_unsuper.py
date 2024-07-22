"""
Build a unsupervised model for indiviudal power and individual beta optimization
Author  : Khin Thandar Kyaw
Reference : DL Framework for Optimization of MISO Downlink Beamforming, TCOM, March 2020
Date    : 8 Nov 2023
Last Modified : 12 Jan 2023
"""

import numpy as np
import tensorflow as tf
import keras
from nn_utils import *
from super_utils import *
from keras import layers
from timer import *

# ------------------------------------
# load and generate simulation data
# ------------------------------------
total_users = total_users()
for user_size in total_users:
  print(f'Total # of Users: {user_size}')
  print_line()
  
  time_list = []
  with Timer() as timer:
  
    antenna_size, _, _, _, _, _, _ = parameters(user_size)
    snr_fixed = fixed_snr()
    
    covariance_complex = np.load(f'test/{user_size}users/cov_test.npy')
    e_max_complex = np.load(f'test/{user_size}users/eMax_test.npy')

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
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    model.summary()
    
    keras.utils.plot_model(model, to_file=f'test/{user_size}users/superModel.png',
                          show_shapes=True, show_layer_names=True, dpi=300)

    # ------------------------------------
    # Save optimal beta and Power values
    # ------------------------------------
    class SavePowerBeta(keras.callbacks.Callback):
      def __init__(self, save_path1, save_path2):
        super(SavePowerBeta, self).__init__()
        self.save_path1 = save_path1
        self.save_path2 = save_path2
        
      def on_test_end(self, logs=None):
        powerBetaModel = keras.Model(inputs=[covariance_stacked_input, 
                                            power_total_input, e_max_stacked_input], 
                                        outputs=[individual_power_output, individual_beta_output])
        powerk, betak = powerBetaModel.predict([covariance_stacked, power_total, e_max_stacked])
        np.save(self.save_path1, powerk)
        np.save(self.save_path2, betak)

    # ------------------------------------
    # Load the Model
    # ------------------------------------
    model.load_weights(f'train/{user_size}users/trainedSuper.h5')
    save_on_eval = SavePowerBeta(f'test/{user_size}users/powerk.npy', 
                              f'test/{user_size}users/betak.npy')
    model.evaluate(x=[covariance_stacked, power_total,
                      identity_matrix, covariance_complex,
                      e_max_complex, e_max_stacked], 
                  y=covariance_complex, # Dummy target
                  batch_size=batch_size, verbose=0, 
                  callbacks=save_on_eval
                  )

    # ------------------------------------
    # Compute the sum rate
    # ------------------------------------
    pk = np.load(f'test/{user_size}users/powerk.npy')
    bk = np.load(f'test/{user_size}users/betak.npy')

    print(f'pk.shape: {pk.shape}')
    print(f'bk.shape: {bk.shape}')

    W = compute_beam([pk, bk, e_max_complex, identity_matrix, covariance_complex])

  time_list.append(timer.elapsed_time)

  ensure_dir(f'Plotting/{user_size}users/')
  np.save(f'Plotting/{user_size}users/beamFromBetaSuper.npy', W)

  #print(f'W.shape: {W.shape}')
  beamNormSquared = compute_norm_squared(W)[0:5]
  print(f'beamNormSquared of Beta NN Model without normalizing: {beamNormSquared}')

  # Check the sum of the norm squared = Power
  print(f'PowerTotal[0: 5]= {power_total[0: 5]}')

  WZ = np.load(f'test/{user_size}users/beamZF.npy')
  print(f'WZ.shape: {WZ.shape}')

  #  Check norm of each beam for ZF = 1
  print('Check if the norm of each beam of ZF = 1')
  beam_Z_norm = compute_norm(WZ)[0:5]
  print(f'beamZNormSquared: {beam_Z_norm}')

  print('Loading...')

  with Timer() as timer:
    rate = []
    for snr in range(-5, 25, 5):
      SNR = np.power(10, np.ones([sample_size, 1]) * snr / 10)
      power = SNR * noise_var_total
      normalized_pk = normalization(pk, power)
      normalized_bk = normalization(bk, power)
      W = compute_beam([normalized_pk, normalized_bk, 
                      e_max_complex, identity_matrix, covariance_complex])
      sum_rate = np.mean(compute_sum_rate(W, covariance_complex))
      rate.append(sum_rate)
  time_list.append(timer.elapsed_time)
    
  np.save(f'test/{user_size}users/timeArraySuper.npy', np.array(time_list))
  np.save(f'Plotting/{user_size}users/sumRateSuper.npy', np.array(rate))
  print("Done!")

