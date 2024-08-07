"""
Timer Process
Author    : Khin Thandar Kyaw
Date      : 12 Jan 2023
"""

import time

################################################################################
def perform_calculations(Covariance, channel_covariance_all):
  U = Covariance.eigen_decomposition_channel_covariance(channel_covariance_all)
  U_star = Covariance.extract_U_star(U)
  U_tilde, size_U_tilde_col = Covariance.construct_U_tilde(U_star)
  E_0 = Covariance.SVD_U_tilde(U_tilde, size_U_tilde_col)
  V_max = Covariance.project_channel_covariance_onto_E_0(E_0, channel_covariance_all)
  W = Covariance.calculate_beamforming_vector(E_0, V_max)
  return U_tilde, W
################################################################################

class Timer:
  def __enter__(self):
    self.start_time = time.time()
    return self
  
  def __exit__(self, *args):
    self.elapsed_time = time.time() - self.start_time