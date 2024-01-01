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


rateNNSuper30 = np.load(f'Plotting/10users/sumRateSuperNt16N30T10.npy')
rateZF30 = np.load(f'Plotting/10users/sumRateZFNt16N30T10.npy')

rateNNSuper60 = np.load(f'Plotting/10users/sumRateSuperNt16N60T10.npy')
rateZF60 = np.load(f'Plotting/10users/sumRateZFNt16N60T10.npy')

rateNNSuper120 = np.load(f'Plotting/10users/sumRateSuperNt16N120T10.npy')
rateZF120 = np.load(f'Plotting/10users/sumRateZFNt16N120T10.npy')


print('Loading...')
linePrint()

plt.figure(figsize=(7, 6))  


# Plot lines
plottingLine(rateZF30, 'ZF-SBF [N = 30]', 'dashed', 'red', '+')
plottingLine(rateNNSuper30, 'Proposed [N = 30]', 'solid', 'red', 'o')

plottingLine(rateZF60, 'ZF-SBF [N = 60]', 'dashed', 'green', '+')
plottingLine(rateNNSuper60, 'Proposed [N = 60]', 'solid', 'green', 'o')

plottingLine(rateZF120, 'ZF-SBF [N = 120]', 'dashed', 'blue', '+')
plottingLine(rateNNSuper120, 'Proposed [N = 120]', 'solid', 'blue', 'o')

# Legend
plt.legend(loc='upper left', ncol=1, fontsize=13)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{tot}}/\sigma_n^2$ (dB)', fontsize=12)
plt.ylabel('Sum rate (bps/Hz)', fontsize=13)

# Title
totalUser = 10
plt.title(r'$N_t$ = {}, $M + K$ = {}'.format(Nt, totalUser), fontsize=13)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.savefig(f'Plotting/sumRateComparison.png')  
plt.close()

print("Done!")


  
