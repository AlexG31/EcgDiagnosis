#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import test_mcmc
import numpy as np
from pymc import MCMC
import scipy.signal as signal
import pdb

M = MCMC(test_mcmc)

M.sample(iter = 10000, burn = 5000, thin = 10)

xlist = M.trace('wave_center')[:]
sigma_list = M.trace('shape_sigma')[:]


center = int(np.mean(xlist))
sigma = np.mean(sigma_list)


plt.figure(2)
# Plot fitting curve
len_sig = test_mcmc.len_sig
len_base = 2 * (test_mcmc.len_sig + abs(center)) + 1
gaussian_base = signal.gaussian(len_base, sigma)
fitting_sig = [gaussian_base[x - center + len_base / 2] for x in xrange(0, len_sig)]
plt.plot(fitting_sig, label = 'fitting curve')
plt.plot(test_mcmc.raw_sig, label = 'ECG')
plt.title('ECG')
plt.legend()
plt.grid(True)
# plt.figure(1)
# plt.hist(xlist)
plt.show()
