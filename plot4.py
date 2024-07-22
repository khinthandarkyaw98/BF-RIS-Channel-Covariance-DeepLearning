"""
Compare the sum rate between the neural network and the Zero-forcing
Author    : Khin Thandar Kyaw
Date : 21 OCT 2023
Last Modified  : 15 Nov 2023
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn_utils import *

# tf_version: 2.15.0
print(tf.__version__)
print("Loading...")

# --------------------------- Start --------------------------------
total_users = ''.join(map(str, total_users()))
Nt, N, _, _, _, _, _ = parameters(6) # 6 is just a placeholder
snr_fixed = fixed_snr()


global_ymin = 0
global_ymax = 60


# load the data
rate_NN_unsuper_case_1 = np.load(f'Plotting/{total_users}users/sumRateSupercase1.npy')
rate_NN_unsuper_case_2 = np.load(f'Plotting/{total_users}users/sumRateSupercase2.npy')

rateWFcase1 = np.load(f'Plotting/{total_users}users/sumRateWFcase1.npy')
rateWFcase2 = np.load(f'Plotting/{total_users}users/sumRateWFcase2.npy')

print('Loading...')
print_line()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.figure(figsize=(7, 6))


# Plot lines
#plottingLine(rateZF60, 'ZF-SBF [N = 60]', 'dotted', 'green', '+')
plot_line(rateWFcase1, r'ZF beam w/ WF pwr [Rank ($R_h$) = 1 for all UEs]', 'dashed', 'blue', 'v')
plot_line(rateWFcase2, r'ZF beam w/ WF pwr [Rank ($R_h$) = 2 for all UEs]', 'dashed', 'red', 'P')
plot_line(rate_NN_unsuper_case_1, r'Proposed [Rank ($R_h$) = 1 for all UEs]', 'solid', 'blue', 'v')
plot_line(rate_NN_unsuper_case_2, r'Proposed [Rank ($R_h$) = 2 for all UEs]', 'solid', 'red', 'P')

# Legend
plt.legend(loc='upper left', ncol=1, fontsize=15)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{T}}/\sigma_n^2$ (dB)', fontsize=16)
plt.ylabel('Approximate sum rate (bps/Hz)', fontsize=16)

# Title
plt.title(r'$N_t$ = {}, $N$ = {}, $M + K$ = {}'.format(Nt, N, total_users), fontsize=16)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
#plt.savefig(f'Plotting/fig4.tiff')  
plt.savefig(f'Plotting/fig4.png')  
plt.savefig('Plotting/fig4.eps', format='eps')
plt.close()

print("Done!")


  
