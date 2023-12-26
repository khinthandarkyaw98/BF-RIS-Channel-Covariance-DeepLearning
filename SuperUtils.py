"""
Utility function of the unsupervised model for beamforming 
with individual power and beta constraints
Author    : Khin Thandar Kyaw
Last Modified : 15 NOV 2023
"""

import numpy as np
import tensorflow as tf
from NNUtils import *

def transPower(temp):
  # (None, userSize), (None, 1)
  individualPower, PowerTotal = temp
  # enforce the constraint that power is positive
  individualPower = tf.nn.softplus(individualPower)
  # (None, userSize, 1)
  individualPower = tf.expand_dims(individualPower, axis=-1)
  # (None, userSize, 1, 1)
  individualPower = tf.expand_dims(individualPower, axis=-1)
  normalizedPower = normalization(individualPower, PowerTotal)
  return normalizedPower

def transBeta(temp):
  # (None, userSize), (None, 1)
  individualBeta, PowerTotal= temp
  # enforce the constraint that beta is positive
  individualBeta = tf.nn.softplus(individualBeta)
  # (None, userSize, 1)
  individualBeta = tf.expand_dims(individualBeta, axis=-1)
  # (None, userSize, 1, 1)
  individualBeta = tf.expand_dims(individualBeta, axis=-1)
  normalizedBeta = normalization(individualBeta, PowerTotal)
  return normalizedBeta

def normalization(NNval, Ptotal):
  NNval = tf.cast(NNval, dtype=tf.float32) # [tf.float64 != tf.float32] error
  Ptotal = tf.cast(Ptotal, dtype=tf.float32) 
  userSize = tf.shape(NNval)[1]
  # (None, 1, 1)
  NNsum = tf.reduce_sum(NNval, axis=1)
  # (None, 1, 1)
  Ptotal = tf.expand_dims(Ptotal, axis=-1)
  # (None, 1, 1)
  division = tf.divide(Ptotal, NNsum)
  # (None, 1, 1, 1)
  division = tf.expand_dims(division, axis=1)
  # (None, userSize, 1, 1)
  division = tf.tile(division, [1, userSize, 1, 1])
  NNstar = tf.multiply(division, NNval) # float
  return NNstar 

def computeSumScaledRh(beta, R):
  userSize = tf.shape(R)[1]
  # (None, antennaSize, antennaSize)
  sumScaledRh = tf.reduce_sum(tf.multiply(beta, R), axis=1)
  # (None, 1, antennaSize, antennaSize)
  sumScaledRh = tf.expand_dims(sumScaledRh, axis=1)
  # (None, userSize, antennaSize, antennaSize)
  sumScaledRh = tf.tile(sumScaledRh, [1, userSize, 1, 1]) # complex
  return sumScaledRh

def computeBeam(temp):
  P, beta, eMax, identity, R = temp
  # to be able to multpliy
  # convert beta to complex
  beta = tf.cast(beta, dtype=tf.complex64)
  # (None, userSize, antennaSize, antennaSize)
  sumScaledRh = computeSumScaledRh(beta, R)
  # (None, userSize, antennaSize, antennaSize)
  identityBetaR = tf.add(identity, sumScaledRh)
  # (None, userSize, antennaSize, antennaSize)
  invIdentityBetaR = tf.linalg.inv(identityBetaR) # complex
  sqrtOfP = tf.cast(tf.sqrt(P), dtype=tf.complex64)
  # (None, userSize, antennaSize, antennaSize)
  # the last two axes are broadcasted to the same shape as invIdentityBetaR
  mul1 = tf.multiply(sqrtOfP, invIdentityBetaR)
  # (None, userSize, antennaSize, 1)
  numerator = tf.matmul(mul1, eMax)
  # (None, userSize, antennaSize, 1)
  mul2 = tf.matmul(invIdentityBetaR, eMax)
  # (None, userSize, 1, antennaSize)
  mul2H = tf.linalg.adjoint(mul2)
  # (None, userSize, 1, 1)
  denominator = tf.sqrt(tf.matmul(mul2H, mul2))
  # (None, userSize, antennaSize, 1)
  W = numerator / denominator # complex
  return W

