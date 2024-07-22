"""
Generate covariance martrix
Author    : Khin Thandar Kyaw
Date      : 31 AUG 2023
Last Modified  : 25 Nov 2023
"""

import numpy as np
from typing import Tuple

class Covariance:
    
  def __init__(self, Nt: int, N: int, total_users: int, M: int, K: int, Lm: np.ndarray, Lk: np.ndarray, Ltotal: np.ndarray) -> None:
    self.Nt = Nt # Total No. of Tx-BS antennas
    self.N = N # Total No. of patches on each IRS
    self.total_users = total_users # M + K
    self.M = M # Total No. of direct Users
    self.K = K # Total No. of IRS-assisted Users
    self.Lm = Lm # Array of No. of paths between BS and each user
    self.Lk = Lk # Array of No. of paths between IRS and each user
    self.Ltotal = Ltotal # Just for the convenience of ZFBF
   
  # ------------------------------------
  # direct user channel
  # ------------------------------------
  def generate_theta(self)-> list:
    return [np.random.uniform(0, 2 * np.pi, size=l).reshape(-1, 1) for l in self.Lm] # column_vectors of different thetas for each user m

  def generate_steering_vectors(self, theta: list)-> list:
    steering_vectors = []
    for theta_m in theta:
      A_m = np.column_stack([(1/np.sqrt(self.Nt)) * np.exp(-1j * 2 * np.pi * np.arange(self.Nt) * np.cos(theta_l)) for theta_l in theta_m])
      steering_vectors.append(A_m)
    return steering_vectors
  
  def generate_channel_covariance(self, steering_vectors: list)-> list:
    channel_covariance = []
    for m in range(self.M):
      A_m = steering_vectors[m]
      A_m_Hermitian = np.conjugate(A_m).T
      R_m = (self.Nt/self.Lm[m]) * np.matmul(A_m, A_m_Hermitian)
      channel_covariance.append(R_m)
      # if m == 0:
      #   print(f'Rank of direct channel_covariance: {np.linalg.matrix_rank(R_m)}')
      #   print(f'\nNorm of direct channel_covariance: {np.linalg.norm(R_m, ord=2)}')
      #   _, _, _, max_eigenvalue = self.eigVecCorrMaxEigVal(R_m)
      #   print(f'Max eigenvalue of direct channel_covariance: {max_eigenvalue}\n')
    return channel_covariance

  ##############################################################
  
  # ------------------------------------
  # IRS-assisted user channel
  # ------------------------------------
  def generate_xi(self)-> np.ndarray:
    return [np.random.uniform(0,  np.pi, size=self.N).reshape(-1, 1) for k in range(self.K)]
  
  def generate_upsilon(self)-> np.ndarray:
    return [np.random.uniform(0, 2 * np.pi, size=self.N).reshape(-1, 1) for k in range(self.K)]
  
  def generate_channelBSIRS(self, xi: np.ndarray, upsilon: np.ndarray)-> list:
    channel_BS_IRS = []
    for k in range(self.K):
      G_k = np.empty((self.Nt, self.N), dtype = np.complex128)
      xi_k = xi[k]
      upsilon_k = upsilon[k]
      for nt in range(self.Nt):
        for n in range(self.N):
          # range starts from 0, so (nt -1) should be nt and (n - 1) should be n
          G_k[nt, n] = np.exp(1j * np.pi * nt * np.sin(xi_k[n]) * np.sin(upsilon_k[n])) * \
          np.exp(-1j * np.pi * n * np.sin(xi_k[n]) * np.sin(upsilon_k[n])) 
      channel_BS_IRS.append(G_k)
      # if k == 0:
      #   print(f'Norm of G_k: {np.linalg.norm(G_k, ord=2)}')
    return channel_BS_IRS
   
  def generate_big_theta(self)-> list:
    big_theta = []
    for _ in range(self.K):
      rand_deg = (1/np.sqrt(self.N)) * np.exp(1j * np.random.uniform(0, 2 * np.pi, self.N))
      diag_matrix = np.diag(rand_deg)
      big_theta.append(diag_matrix)  
    return big_theta  
      
  def generate_phi(self)-> list:
    return [np.random.uniform(0, 2 * np.pi, size=l).reshape(-1, 1) for l in self.Lk] # column_vectors of different phis for each user k
  
  def generate_steering_vectors_irs(self, phi: list)-> list:
    steering_vectors_irs = []
    for phi_k in phi:
      B_k = np.column_stack([(1/np.sqrt(self.N)) * np.exp(-1j * 2 * np.pi * np.arange(self.N) * np.cos(phi_n)) for phi_n in phi_k])
      steering_vectors_irs.append(B_k)
    return steering_vectors_irs
  
  def generate_channel_covariance_irs(self, steering_vectors_irs: list)-> list:
    channel_covariance_irs = []
    for k in range(self.K):
      B_k = steering_vectors_irs[k]
      B_k_Hermitian = np.conjugate(B_k).T
      R_g = (self.N/self.Lk[k]) * np.matmul(B_k, B_k_Hermitian)
      channel_covariance_irs.append(R_g)
      # if k == 0:
      #   print(f'Rank of R_g: {np.linalg.matrix_rank(R_g)}')
      #   print(f'Norm of R_g: {np.linalg.norm(R_g, ord=2)}')
      #   _, _, _, max_eigenvalue = self.eigVecCorrMaxEigVal(R_g)
      #   print(f'Max eigenvalue of R_g: {max_eigenvalue}\n')
    return channel_covariance_irs
  
  def generate_composite_channel_covariance(self, channelBSIRS: list, big_theta: list, channel_covariance_irs: list, channel_covaraince: list)-> list:
    for k in range(self.K):
      G_k = channelBSIRS[k]
      big_theta_k = big_theta[k]
      R_g = channel_covariance_irs[k]
      G_k_Hermitian = np.conjugate(G_k).T
      big_theta_k_Hermitian = np.conjugate(big_theta_k).T 
      mul_1 = np.matmul(G_k, big_theta_k)
      mul_2 = np.matmul(mul_1, R_g)
      mul_3 = np.matmul(big_theta_k_Hermitian, G_k_Hermitian)
      R_h_k = np.matmul(mul_2, mul_3)
      channel_covaraince.append(R_h_k) # included direct channel covariance
      # if k == 0:
      #   print(f'Rank of composite channel_covariance: {np.linalg.matrix_rank(R_h_k)}')
      #   print(f'Norm of composite channel_covariance: {np.linalg.norm(R_h_k, ord=2)}')
      #   _, _, _, max_eigenvalue = self.eigVecCorrMaxEigVal(R_h_k)
      #   print(f'Max eigenvalue of composite channel_covariance: {max_eigenvalue}\n')
    return channel_covaraince
  
  ##############################################################
  
  # ------------------------------------
  # Total channel covariance
  # ------------------------------------
  # m and k in the followings are just m and k, not related to M and K
  
  def eig_vec_corr_max_eig_val(self, matrix: list)-> Tuple [list, list, list]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1] # sort in the descending order
    sorted_eigenvalues = eigenvalues[sorted_indices] # reorder the eigenvalues
    # sorted eigenvectors corresponding to the largest eigenvalue
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # select the largest eigenvector ( leftmost column )
    ekMax = sorted_eigenvectors[:, 0].reshape(-1, 1)
    max_eigenvalue = sorted_eigenvalues[0]
    return ekMax, sorted_eigenvalues, sorted_eigenvectors, max_eigenvalue
  
  def e_max(self, channel_covariance: list):
    e_max = []
    for m in range(self.total_users):
      R_m = channel_covariance[m]
      e_m_max, _, _, _ = self.eig_vec_corr_max_eig_val(R_m)
      e_max.append(e_m_max)
    return e_max
  
  def eigen_decomposition_channel_covariance(self, channel_covariance: list)-> list:
    U = []
    U_Hermitian = []
    Lambda = []
    for m in range(self.total_users):
      R_m = channel_covariance[m]
      _, sorted_eigenvalues, sorted_eigenvectors, _ = self.eig_vec_corr_max_eig_val(R_m)
      U_m = sorted_eigenvectors
      U_m_Hermitian = np.conjugate(sorted_eigenvectors).T
      U.append(U_m)
      U_Hermitian.append(U_m_Hermitian)
      Lambda_m = np.diag(sorted_eigenvalues)
      Lambda.append(Lambda_m)
    return U
  
  def extract_U_star(self, U:list)-> list:
    U_star = []
    for m in range(self.total_users):
      U_star_m = U[m][:, :self.Ltotal[m]] # sorted eigenvectors [just take out in dimension (Nt, Lm)]
      U_star.append(U_star_m)
    return U_star
  
  def construct_U_tilde(self, U_star: list)-> Tuple[list, int]:
    U_tilde = []
    size_U_tilde_col = []
    for k in range(self.total_users):
      indices_without_k = [m for m in range(self.total_users) if m!=k] # omit k
      U_star_without_k = [U_star[idx] for idx in indices_without_k]
      U_tilde_k = np.hstack(U_star_without_k) # stack arrays horrizontally to form matrix : U_tilde2 = [U_star0, U_star1] for M = 3
      size_U_tilde_k_col = U_tilde_k.shape[1]
      size_U_tilde_col.append(size_U_tilde_k_col)
      U_tilde.append(U_tilde_k)
    return U_tilde, size_U_tilde_col
  
  def SVD_U_tilde(self, U_tilde: list, size_U_tilde_k_col: list)-> list:
    Sigma = []
    E_1 = []
    E_0 = []
    for m in range(self.total_users):
      E_k, Sigma_k, V_h_k = np.linalg.svd(U_tilde[m])
      Sigma.append(Sigma_k)
      E_k_1 = E_k[:, :size_U_tilde_k_col[m]]
      E_k_0 = E_k[:, size_U_tilde_k_col[m]:]
      E_1.append(E_k_1)
      E_0.append(E_k_0)
    return E_0
  
  def project_channel_covariance_onto_E_0(self, E_0: list, channel_covariance: list)-> list:
    V_max = []
    for m in range(self.total_users):
      E_m_0_h = np.conjugate(E_0[m]).T
      resulted_matrix = E_m_0_h @ channel_covariance[m] @ E_0[m]
      V_m_max, _, _, _ = self.eig_vec_corr_max_eig_val(resulted_matrix)
      V_max.append(V_m_max)
    return V_max
  
  def calculate_beamforming_vector(self, E_0: list, V_max: list)-> list:
    w = []
    for m in range(self.total_users):
      w_m = E_0[m] @ V_max[m]
      w.append(w_m)
    return w
  
  def check_ZFBF_condition(self, U_tilde: list, w:list)-> list:
    ZFBF_res = []
    abs_ZFBF_res = []
    for m in range(self.total_users):
      U_tilde_m_Hermitian = np.conjugate(U_tilde[m]).T
      output = U_tilde_m_Hermitian @ w[m]
      ZFBF_res.append(output)
      abs_output = np.absolute(output)
      abs_ZFBF_res.append(abs_output)
    return abs_ZFBF_res
    
    