"""
Utility functions for the neural networks
Author    : Khin Thandar Kyaw
Last Modified : 25 NOV 2023
"""

import random
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from textwrap import wrap

def print_line():
  print('=' * 50)

def fixed_snr():
  snr = [-5, 0, 5, 10, 15, 20] # dB
  return snr

def total_users():
  total_users = [6, 8, 10]  # figure 2 and figure 3
  #totalUsers = [8] # figure 4
  return total_users

def parameters(total_user):
  #Nt = 10 # Nt = antenna_size
  Nt = 16
  N = 30 # N = No. of patches on each IRS # figure 2, figure 3
  #N = 60 # N = No. of patches on each IRS # figure 4
  #N = 120 # N = No. of patches on each IRS # figure 3
  #start = totalUser - 3
  #M =  random.randint(start, totalUser - 1)
  end = total_user
  M =  random.randint(0, end) # M = direct user_size
  K = total_user - M # K = IRS-assisted user_size
  while True:
    #Lm = np.random.randint(2, 3, size=M)
    Lm = np.random.randint(1, 2, size=M) # Lm = path between BS and user
    #Lk = np.random.randint(2, 3, size=K)
    Lk = np.random.randint(1, 2, size=K) # Lk = path between IRS and user
    # Ltotal = np.random.randint(1, 2, size=totalUser) # Ltotal = path between BS/IRS and user
    Ltotal = np.concatenate((Lm, Lk))
    if np.sum(Lm) + np.sum(Lk) <= Nt:
      break
    else:
      print('Lm + Lk > Nt must not be!')
      return 0, 0, 0, 0, 0, 0, 0
  return Nt, N, M, K, Lm, Lk, Ltotal

# ------------------------------------
# save the data
# ------------------------------------
def ensure_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def stacking(complex_data):
  real_data = np.real(complex_data)
  img_data = np.imag(complex_data)
  # axis = 2
  # (sampleSize, userSize, real/imag, antennaSize, antennaSize)
  data_stacked = np.stack([real_data, img_data], axis=2)
  return data_stacked

def data_preparation(matrix):
  batch_size = 32
  sample_size = matrix.shape[0] # this is the number of samples in the train or test set
  
  # --------------stacking-----------------
  covariance_stacked = stacking(matrix)
  
  # Generate the fixed SNR associated with all samples
  #snr = fiexdSNR()
  snr_total = np.power(10, np.random.randint(-5, 25, [sample_size, 1]) / 10)
  #SNRTotal = np.power(10, np.ones([sampleSize, 1]) * snr / 10)
  
  # noise variance
  noise_var_total = np.ones([sample_size, 1])
  
  # Total Power
  power_total = snr_total * noise_var_total
  return batch_size, sample_size, covariance_stacked, snr_total, noise_var_total, power_total

def compute_sum_rate(W, R, alpha=1.0):
  # batch_size is the number of samples in the batch
  batch_size = tf.shape(R)[0]
  user_size = tf.shape(R)[1]
  W_H = tf.linalg.adjoint(W)
  # (batchSize, userSize, 1, 1)
  numerator = tf.math.real(tf.einsum('bmqk, bmkl, bmlp->bmqp', W_H, R, W))
  # (batchSize, userSize, userSize, 1, 1)
  all_interferences = tf.math.real(tf.einsum('biqk, bmkl, bilp->bimqp', W_H, R, W))
  # e.g if user_size = 2, then mask = [[1, 0], [0, 1]]
  mask = tf.eye(user_size, dtype=tf.float32)
  # Add batch Size dimension for more than one sample_size
  # e.g. [[1, 0], [0, 1]] -> [[[1, 0], [0,1]]]
  #  (1, userSize, userSize)
  mask = tf.expand_dims(mask, axis=0)
  # Replicate for each sample
  # e.g. [[[1, 0], [0, 1]]] -> [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
  # (batchSize, userSize, userSize)
  mask = tf.tile(mask, [batch_size, 1, 1])
  # Reshape for broadcasting to compensate bimqp
  mask = tf.reshape(mask, [batch_size, user_size, user_size, 1, 1])
  # Null out the diagonal elements
  mask = tf.cast(mask, all_interferences.dtype) # plot.py if not casted
  all_interferences *= (1 - mask)
  #  (batchSize, 1)
  interference = tf.math.real(tf.reduce_sum(all_interferences, axis=1))
  alphaSquared = tf.square(alpha)
  factor = 1.0 / alphaSquared
  denominator = interference + factor
  SINR = tf.cast(tf.squeeze(numerator / denominator), dtype=tf.float32)
  logBase = tf.math.log(1.0 + SINR) / tf.math.log(2.0)
  sum_rate = tf.reduce_sum(logBase, axis=1)
  return sum_rate

def compute_norm_squared(beam):
  beam_H = tf.linalg.adjoint(beam)
  # (batch_size,)
  beam_norm_squared = tf.reduce_sum(tf.math.real(tf.einsum('bmqk, bmkq->bm', beam_H, beam)), axis=1)
  # (batch_size, 1) to match the dimension of Power
  beam_norm_squared = tf.expand_dims(beam_norm_squared, axis=-1)
  return beam_norm_squared

def compute_norm(beam):
  w_H = tf.linalg.adjoint(beam)
  w_norm = tf.sqrt(tf.einsum('bmqk, bmkq->bm', w_H, beam))
  return w_norm

# parameters returned by the model are tensors
def loss_func_unsuper(temp):
  R, W = temp
  sum_rate = compute_sum_rate(W, R)
  loss = tf.cast( 0.0 - sum_rate , dtype=tf.float32)
  return tf.reduce_mean(loss)
  
def normalize_beam(beam, power):
  w_norm_squared = compute_norm_squared(beam)
  alpha = tf.sqrt(power/ w_norm_squared)
  alpha = tf.expand_dims(alpha, axis=-1)
  alpha = tf.expand_dims(alpha, axis=-1)
  beam_real = tf.math.real(beam)
  beam_img = tf.math.imag(beam)
  w_normalized_real = alpha * beam_real
  w_normalized_img = alpha * beam_img
  w_normalized = tf.complex(w_normalized_real, w_normalized_img)
  return w_normalized

def loss_curve(history, user_size, type, snr_fixed):
  plt.figure(figsize=(8, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  title = f'Training and Validation Loss over Epochs for {user_size} Users of {type} NN model at SNR = {snr_fixed} dB'
  plt.title("\n".join(wrap(title, 60))) # Wrap title at 60 characters
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()
  plt.grid()
  
def plot_line(data, label, line_style, color, marker):
  plt.plot(range(-5, 25, 5), data, label=label, linestyle=line_style, color=color, marker=marker, markersize=8)
  



