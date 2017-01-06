#encoding:utf8

# MCMC Model
import os
import sys
import pymc
import matplotlib.pyplot as plt
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np
import json

# Load ECG segment array
with open('./segment.json', 'r') as fin:
    raw_sig = json.load(fin)
    raw_sig = np.array(raw_sig)
    # Normalize
    min_val = np.min(raw_sig)
    max_val = np.max(raw_sig)
    raw_sig -= min_val
    raw_sig /= (max_val - min_val)


# Length of the ECG segment 
len_sig = raw_sig.size

wave_center = DiscreteUniform('wave_center', lower=0, upper=len_sig, doc='WaveCetner[index]')

shape_sigma = pymc.Normal('shape_sigma', 7, 10000)


@deterministic(plot=False)
def wave_shell(ct=wave_center, sigma=shape_sigma):
    ''' Concatenate wave.'''

    # Make gaussian base len odd
    len_gaussian_base = 2 * max(ct, len_sig - ct) + 1
    gaussian_base = signal.gaussian(len_gaussian_base, sigma)
    out = gaussian_base[0 - ct + len_gaussian_base / 2:len_sig - ct + len_gaussian_base / 2]
    return out



ecg = pymc.Normal('ecg', mu=wave_shell, tau = 4e4, value=raw_sig, observed=True)


def test():
    '''Compare Gaussian Function.'''

    import scipy.signal as signal
    xlist = signal.gaussian(100, 7)
    # plt.hist(xlist)
    plt.plot(xlist)
    plt.title('ECG Segment')
    plt.show()

# test()
