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


global_ymin = 0
global_ymax = 100


# load the data
rateNNSuper30 = np.load(f'Plotting/10users/sumRateSuperN30model.npy')

rateNNSuper60 = np.load(f'Plotting/10users/sumRateSuperN60model.npy')
rateZF60 = np.load(f'Plotting/10users/sumRateZFN60.npy')

rateNNSuper120 = np.load(f'Plotting/10users/sumRateSuperN120model.npy')

print('Loading...')
linePrint()

plt.figure(figsize=(7, 6))  


# Plot lines
plottingLine(rateZF60, 'ZF-SBF [N = 60]', 'dashed', 'green', '+')
plottingLine(rateNNSuper120, 'Proposed [N = 120]', 'solid', 'blue', '|')

plottingLine(rateNNSuper60, 'Proposed [N = 60]', 'solid', 'green', 'x')

plottingLine(rateNNSuper30, 'Proposed [N = 30]', 'solid', 'red', '1')


# Legend
plt.legend(loc='upper left', ncol=1, fontsize=13)
plt.ylim([global_ymin, global_ymax])

# Axes labels
plt.rc('text', usetex=True)
plt.xlabel(r'$P_{\mathrm{tot}}/\sigma_n^2$ (dB)', fontsize=12)
plt.ylabel('Sum rate (bps/Hz)', fontsize=13)

# Title
Nt = 16
totalUsers = 10
plt.title(r'$N_t$ = {}, $M + K$ = {}'.format(Nt, totalUsers), fontsize=13)

plt.grid(True) 
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.savefig(f'Plotting/sumRateComparisonTrainedAtN60TestedwithDiffN.png')  
plt.close()

print("Done!")


  
