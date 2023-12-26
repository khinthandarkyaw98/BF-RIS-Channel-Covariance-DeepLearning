"""
Compare the sum rate between the neural network and the Zero-forcing
Author    : Khin Thandar Kyaw
Date : 21 OCT 2023
Last Modified  : 15 Nov 2023
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from NNUtils import *


# tf_version: 2.15.0
print(tf.__version__)
print("Loading...")

# --------------------------- Start --------------------------------
snrFixed = fiexdSNR()
Nt, N, _, _, _, _, _ = parameters(6) # 6 is just a placeholder

global_ymin = 0
global_ymax = 100


# load the data

rateNNSuper10 = np.load(f'Plotting/10users/sumRateSuper.npy')
rateZF10 = np.load(f'Plotting/10users/sumRateZF.npy')

print('Loading...')
linePrint()

plt.figure(figsize=(7, 6))  


# Plot lines

plottingLine(rateZF10, 'ZF-SBF [M+K=10]', 'dashed', 'green', '+')
plottingLine(rateNNSuper10, 'Proposed [M+K=10]', 'solid', 'green', 'o')

# Legend
plt.legend(loc='upper left', ncol=1, fontsize=13)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{tot}}/\sigma_n^2$ (dB)', fontsize=12)
plt.ylabel('Sum rate (bps/Hz)', fontsize=13)

# Title
plt.title(r'$N_t$ = {}, N = {}'.format(Nt, N), fontsize=13)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.savefig(f'Plotting/sumRateComparison.png')  
plt.close()

print("Done!")


  
