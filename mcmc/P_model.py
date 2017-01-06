#encoding:utf8

# MCMC Model
import os
import sys
import pymc
import math
import matplotlib.pyplot as plt
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np
import json

# Load ECG segment array
with open('./tmp_signal.json', 'r') as fin:
    fs = 250.0
    raw_sig = json.load(fin)
    raw_sig = np.array(raw_sig, dtype = np.float32)
    # Normalize
    min_val = np.min(raw_sig)
    max_val = np.max(raw_sig)
    if max_val - min_val > 1e-6:
        raw_sig -= min_val
        raw_sig /= float(max_val - min_val)

    # plt.plot(raw_sig)
    # plt.title('Input ECG')
    # plt.show()


# Hermit functions
def HermitFunction(level, size):
    '''Return hermit function for P wave.'''
    size = int(size)
    if size < 0:
        raise Exception('Size must be greater or equal to zero!')

    def He0(x):
        return 1.0
    def He1(x):
        return x
    def He2(x):
        return x * x - 1
    def He3(x):
        return x ** 3.0 - 3.0 * x
    def He4(x):
        return x ** 4.0 - 6.0 * x ** 2.0 + 3.0
    def He5(x):
        return x ** 5.0 - 10.0 * x ** 3.0 + 15.0 * x
    # Mapping wave_width to range [-3,3]
    # wave_width = 50 * fs / 250.0
    x_ratio = 6.0 / size 
    if level == 0:
        hermit = He0
    elif level == 1:
        hermit = He1
    elif level == 2:
        hermit = He2
    elif level == 3:
        hermit = He3
    elif level == 4:
        hermit = He4
    elif level == 5:
        hermit = He5

    data = [hermit((x - size / 2) * x_ratio) / 20.0 for x in xrange(0, size)]
    
    return np.array(data)


def GetBaselineMatrix(signal_length, fs):
    '''Get baseline coefficient matrix.(0.5Hz~1Hz)'''
    mat = [[1.0,] * signal_length]
    # 0.5Hz
    sin_list = [math.sin(x / fs * math.pi) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    cos_list = [math.cos(x / fs * math.pi) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    # 1Hz
    sin_list = [math.sin(x / fs * math.pi * 2.0) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    cos_list = [math.cos(x / fs * math.pi * 2.0) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    return np.array(mat)
    

# plt.plot(raw_sig)
# plt.plot(HermitFunction(5, len(raw_sig)))
# plt.show()

# Length of the ECG segment 
len_sig = raw_sig.size

# wave_center = DiscreteUniform('wave_center', lower=0, upper=len_sig, doc='WaveCetner[index]')

hc0 = pymc.Normal('hc0', 1, 0.25)
hc1 = pymc.Normal('hc1', 1, 0.25)
hc2 = pymc.Normal('hc2', 1, 0.25)
hc3 = pymc.Normal('hc3', 1, 0.25)
hc4 = pymc.Normal('hc4', 1, 0.25)
hc5 = pymc.Normal('hc5', 1, 0.25)


baseline_coefs = list()
for ind in xrange(0, 5):
    baseline_coefs.append(pymc.Normal('bs%d' % ind, 0, 1.0))
# P_pos = pymc.Normal('P_pos', 0.6 * len_sig, 36.0 / (len_sig ** 2))
P_pos = DiscreteUniform('P_pos', lower=25, upper=len_sig - 5, doc='WaveCetner[index]')
P_amp = pymc.Normal('P_amp', 0, 0.25)

# @deterministic(plot=False)
# def wave_shell(hc0=hc0,
        # hc1=hc1,
        # hc2=hc2,
        # hc3=hc3,
        # hc4=hc4,
        # hc5=hc5,
        # bs0 = bs0,
        # bs1 = bs1,
        # bs2 = bs2,
        # bs3 = bs3,
        # bs4 = bs4,
        # pos = P_pos,
        # amp = P_amp,
        # ):
    # ''' Concatenate wave.'''

    # p_shape_coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
    # baseline_kx = [bs0, bs1,bs2,bs3, bs4,]

    # out = np.zeros(len_sig,)
    # # add baseline
    # baseline_mat = GetBaselineMatrix(len_sig, fs)
    # for baseline_list, kx in zip(baseline_mat, baseline_kx):
        # out += baseline_list * kx

    # p_wave_length = 50.0 * fs / 250.0
    # p_wave_list = np.zeros(p_wave_length,)
    # for level, coef in zip(xrange(0,6), p_shape_coefs):
        # p_wave_list += HermitFunction(level, p_wave_length) * coef * amp
    # for p_ind, p_val in enumerate(p_wave_list):
        # sig_ind = p_ind - p_wave_length / 2.0 + pos
        # sig_ind = int(sig_ind)
        # # Out of scope
        # if sig_ind >= len_sig or sig_ind < 0:
            # continue
        # out[sig_ind] += p_val
        
    # return out

@deterministic(plot=False)
def wave_shell(hc0=hc0,
        hc1=hc1,
        hc2=hc2,
        hc3=hc3,
        hc4=hc4,
        hc5=hc5,
        bs_list = baseline_coefs,
        pos = P_pos,
        amp = P_amp,
        ):
    ''' Concatenate wave.'''

    p_shape_coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
    baseline_kx = bs_list

    out = np.zeros(len_sig,)
    # add baseline
    baseline_mat = GetBaselineMatrix(len_sig, fs)
    for baseline_list, kx in zip(baseline_mat, baseline_kx):
        out += baseline_list * kx

    p_wave_length = 50.0 * fs / 250.0
    p_wave_list = np.zeros(p_wave_length,)
    for level, coef in zip(xrange(0,6), p_shape_coefs):
        p_wave_list += HermitFunction(level, p_wave_length) * coef * amp
    for p_ind, p_val in enumerate(p_wave_list):
        sig_ind = p_ind - p_wave_length / 2.0 + pos
        sig_ind = int(sig_ind)
        # Out of scope
        if sig_ind >= len_sig or sig_ind < 0:
            continue
        out[sig_ind] += p_val
        
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
