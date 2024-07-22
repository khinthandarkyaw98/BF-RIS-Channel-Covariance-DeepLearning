"""
Utility function of the unsupervised model for beamforming 
with individual power and beta constraints
Author    : Khin Thandar Kyaw
Last Modified : 15 NOV 2023
"""

import numpy as np
import tensorflow as tf
from nn_utils import *

def trans_power(temp):
  # (None, userSize), (None, 1)
  individual_power, power_total = temp
  # enforce the constraint that power is positive
  individual_power = tf.nn.softplus(individual_power)
  # (None, userSize, 1)
  individual_power = tf.expand_dims(individual_power, axis=-1)
  # (None, userSize, 1, 1)
  individual_power = tf.expand_dims(individual_power, axis=-1)
  normalized_power = normalization(individual_power, power_total)
  return normalized_power

def trans_Beta(temp):
  # (None, userSize), (None, 1)
  individual_Beta, power_total= temp
  # enforce the constraint that beta is positive
  individual_Beta = tf.nn.softplus(individual_Beta)
  # (None, userSize, 1)
  individual_Beta = tf.expand_dims(individual_Beta, axis=-1)
  # (None, userSize, 1, 1)
  individual_Beta = tf.expand_dims(individual_Beta, axis=-1)
  normalized_Beta = normalization(individual_Beta, power_total)
  return normalized_Beta

def normalization(NN_val, p_total):
  NN_val = tf.cast(NN_val, dtype=tf.float32) # [tf.float64 != tf.float32] error
  p_total = tf.cast(p_total, dtype=tf.float32) 
  user_size = tf.shape(NN_val)[1]
  # (None, 1, 1)
  NN_sum = tf.reduce_sum(NN_val, axis=1)
  # (None, 1, 1)
  p_total = tf.expand_dims(p_total, axis=-1)
  # (None, 1, 1)
  division = tf.divide(p_total, NN_sum)
  # (None, 1, 1, 1)
  division = tf.expand_dims(division, axis=1)
  # (None, userSize, 1, 1)
  division = tf.tile(division, [1, user_size, 1, 1])
  NN_star = tf.multiply(division, NN_val) # float
  return NN_star 

def compute_sum_scaled_Rh(beta, R):
  user_size = tf.shape(R)[1]
  # (None, antennaSize, antennaSize)
  sum_scaled_Rh = tf.reduce_sum(tf.multiply(beta, R), axis=1)
  # (None, 1, antennaSize, antennaSize)
  sum_scaled_Rh = tf.expand_dims(sum_scaled_Rh, axis=1)
  # (None, userSize, antennaSize, antennaSize)
  sum_scaled_Rh = tf.tile(sum_scaled_Rh, [1, user_size, 1, 1]) # complex
  return sum_scaled_Rh

def compute_beam(temp):
  P, beta, e_max, identity, R = temp
  # to be able to multpliy
  # convert beta to complex
  beta = tf.cast(beta, dtype=tf.complex64)
  # (None, userSize, antennaSize, antennaSize)
  sum_scaled_Rh = compute_sum_scaled_Rh(beta, R)
  # (None, userSize, antennaSize, antennaSize)
  identity_Beta_R = tf.add(identity, sum_scaled_Rh)
  # (None, userSize, antennaSize, antennaSize)
  inv_identity_Beta_R = tf.linalg.inv(identity_Beta_R) # complex
  sqrt_P = tf.cast(tf.sqrt(P), dtype=tf.complex64)
  # (None, userSize, antennaSize, antennaSize)
  # the last two axes are broadcasted to the same shape as invIdentityBetaR
  mul1 = tf.multiply(sqrt_P, inv_identity_Beta_R)
  # (None, userSize, antennaSize, 1)
  numerator = tf.matmul(mul1, e_max)
  # (None, userSize, antennaSize, 1)
  mul2 = tf.matmul(inv_identity_Beta_R, e_max)
  # (None, userSize, 1, antennaSize)
  mul2_H = tf.linalg.adjoint(mul2)
  # (None, userSize, 1, 1)
  denominator = tf.sqrt(tf.matmul(mul2_H, mul2))
  # (None, userSize, antennaSize, 1)
  W = numerator / denominator # complex
  return W

