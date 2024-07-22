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
snr_fixed = fixed_snr()
Nt, N, _, _, _, _, _ = parameters(6) # 6 is just a placeholder

global_ymin = 0
global_ymax = 60


# load the data
rate_NN_unsuper_6 = np.load(f'Plotting/6users/sumRateSuper.npy')
#rateZF6 = np.load(f'Plotting/6users/sumRateZF.npy')
rate_WF_6 = np.load(f'Plotting/6users/sumRateWF.npy')

rate_NN_unsuper_8 = np.load(f'Plotting/8users/sumRateSuper.npy')
#rateZF8 = np.load(f'Plotting/8users/sumRateZF.npy')
rate_WF_8 = np.load(f'Plotting/8users/sumRateWF.npy')

rate_NN_unsuper_10 = np.load(f'Plotting/10users/sumRateSuper.npy')
#rateZF10 = np.load(f'Plotting/10users/sumRateZF.npy')
rate_WF_10 = np.load(f'Plotting/10users/sumRateWF.npy')

print('Loading...')
print_line()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.figure(figsize=(7, 6))


# Plot lines
#plottingLine(rateZF6, 'ZF-SBF [M+K=6]', 'dotted', 'red', '+')
plot_line(rate_NN_unsuper_6, 'Proposed [M+K=6]', 'solid', 'red', 'P')
plot_line(rate_WF_6, 'ZF beam w/ WF pwr [M+K=6]', 'dashed', 'red', 'P')

#plottingLine(rateZF8, 'ZF-SBF [M+K=8]', 'dotted', 'green', '+')
plot_line(rate_NN_unsuper_8, 'Proposed [M+K=8]', 'solid', 'green', 'v')
plot_line(rate_WF_8, 'ZF beam w/ WF pwr [M+K=8]', 'dashed', 'green', 'v')

#plottingLine(rateZF10, 'ZF-SBF [M+K=10]', 'dotted', 'blue', '+')
plot_line(rate_NN_unsuper_10, 'Proposed [M+K=10]', 'solid', 'blue', 'd')
plot_line(rate_WF_10, 'ZF beam w/ WF pwr [M+K=10]', 'dashed', 'blue', 'd')

# Legend
plt.legend(loc='upper left', ncol=1, fontsize=16)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{T}}/\sigma_n^2$ (dB)', fontsize=16)
plt.ylabel('Approximate sum rate (bps/Hz)', fontsize=16)

# Title
plt.title(r'$N_t$ = {}, N = {}'.format(Nt, N), fontsize=16)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
#plt.savefig('Plotting/fig2.tiff')
plt.savefig('Plotting/fig2.png')  
plt.savefig('Plotting/fig2.eps', format='eps')
plt.close()

print("Done!")


  
