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

def linePrint():
  print('=' * 50)

def fiexdSNR():
  snr = [-5, 0, 5, 10, 15, 20] # dB
  return snr

def totalUsersFunc():
  #totalUsers = [6, 8, 10]  # figure 2 and figure 3
  totalUsers = [8] # figure 4
  return totalUsers

def parameters(totalUser):
  #Nt = 10 # Nt = antenna_size
  Nt = 16
  N = 30 # N = No. of patches on each IRS # figure 2, figure 3
  #N = 60 # N = No. of patches on each IRS # figure 4
  #N = 120 # N = No. of patches on each IRS # figure 3
  #start = totalUser - 3
  #M =  random.randint(start, totalUser - 1)
  end = totalUser
  M =  random.randint(0, end) # M = direct user_size
  K = totalUser - M # K = IRS-assisted user_size
  while True:
    #Lm = np.random.randint(2, 3, size=M)
    Lm = np.random.randint(1, 2, size=M) # Lm = path between BS and user
    #Lk = np.random.randint(2, 3, size=K)
    Lk = np.random.randint(1, 2, size=K) # Lk = path between IRS and user
    Ltotal = np.random.randint(1, 2, size=totalUser) # Ltotal = path between BS/IRS and user
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

def stacking(complexData):
  realData = np.real(complexData)
  imagData = np.imag(complexData)
  # axis = 2
  # (sampleSize, userSize, real/imag, antennaSize, antennaSize)
  dataStacked = np.stack([realData, imagData], axis=2)
  return dataStacked

def dataPreparation(matrix):
  batchSize = 32
  sampleSize = matrix.shape[0] # this is the number of samples in the train or test set
  
  # --------------stacking-----------------
  covarianceStacked = stacking(matrix)
  
  # Generate the fixed SNR associated with all samples
  #snr = fiexdSNR()
  SNRTotal = np.power(10, np.random.randint(-5, 25, [sampleSize, 1]) / 10)
  #SNRTotal = np.power(10, np.ones([sampleSize, 1]) * snr / 10)
  
  # noise variance
  NoiseVarTotal = np.ones([sampleSize, 1])
  
  # Total Power
  PowerTotal = SNRTotal * NoiseVarTotal
  return batchSize, sampleSize, covarianceStacked, SNRTotal, NoiseVarTotal, PowerTotal

def computeSumRate(W, R, alpha=1.0):
  # batch_size is the number of samples in the batch
  batchSize = tf.shape(R)[0]
  userSize = tf.shape(R)[1]
  W_H = tf.linalg.adjoint(W)
  # (batchSize, userSize, 1, 1)
  numerator = tf.math.real(tf.einsum('bmqk, bmkl, bmlp->bmqp', W_H, R, W))
  # (batchSize, userSize, userSize, 1, 1)
  all_interferences = tf.math.real(tf.einsum('biqk, bmkl, bilp->bimqp', W_H, R, W))
  # e.g if user_size = 2, then mask = [[1, 0], [0, 1]]
  mask = tf.eye(userSize, dtype=tf.float32)
  # Add batch Size dimension for more than one sample_size
  # e.g. [[1, 0], [0, 1]] -> [[[1, 0], [0,1]]]
  #  (1, userSize, userSize)
  mask = tf.expand_dims(mask, axis=0)
  # Replicate for each sample
  # e.g. [[[1, 0], [0, 1]]] -> [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
  # (batchSize, userSize, userSize)
  mask = tf.tile(mask, [batchSize, 1, 1])
  # Reshape for broadcasting to compensate bimqp
  mask = tf.reshape(mask, [batchSize, userSize, userSize, 1, 1])
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

def computeNormSquared(beam):
  beam_H = tf.linalg.adjoint(beam)
  # (batch_size,)
  beamNormSquared = tf.reduce_sum(tf.math.real(tf.einsum('bmqk, bmkq->bm', beam_H, beam)), axis=1)
  # (batch_size, 1) to match the dimension of Power
  beamNormSquared = tf.expand_dims(beamNormSquared, axis=-1)
  return beamNormSquared

def computeNorm(beam):
  WH = tf.linalg.adjoint(beam)
  WNorm = tf.sqrt(tf.einsum('bmqk, bmkq->bm', WH, beam))
  return WNorm

# parameters returned by the model are tensors
def lossFuncSuper(temp):
  R, W = temp
  sumRate = computeSumRate(W, R)
  loss = tf.cast( 0.0 - sumRate , dtype=tf.float32)
  return tf.reduce_mean(loss)
  
def normalizedBeam(beam, Power):
  WNormSquared = computeNormSquared(beam)
  alpha = tf.sqrt(Power/ WNormSquared)
  alpha = tf.expand_dims(alpha, axis=-1)
  alpha = tf.expand_dims(alpha, axis=-1)
  beamReal = tf.math.real(beam)
  beamImag = tf.math.imag(beam)
  WNormalizedReal = alpha * beamReal
  WNormalizedImag = alpha * beamImag
  WNormalized = tf.complex(WNormalizedReal, WNormalizedImag)
  return WNormalized

def lossCurve(history, userSize, type, snrFixed):
  plt.figure(figsize=(8, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  title = f'Training and Validation Loss over Epochs for {userSize} Users of {type} NN model at SNR = {snrFixed} dB'
  plt.title("\n".join(wrap(title, 60))) # Wrap title at 60 characters
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()
  plt.grid()
  
def plottingLine(Data, Label, LineStyle, Color, Marker):
  plt.plot(range(-5, 25, 5), Data, label=Label, linestyle=LineStyle, color=Color, marker=Marker, markersize=8)
  



