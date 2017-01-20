#encoding:utf8

# MCMC Model
import os
import sys
import pymc
import math
import scipy.signal
import matplotlib.pyplot as plt
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np
import json

# Hermit functions
HermitFunction_max_level = 4
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
    def He6(x):
        return x ** 6.0 - 15.0 * x ** 4.0 + 45.0 * x ** 2 - 15.0
    def He7(x):
        return x ** 7.0 - 21.0 * x ** 5.0 + 105.0 * x ** 3 - 105.0 * x
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
    elif level == 6:
        hermit = He6
    elif level == 7:
        hermit = He7

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
def GetGaussianPwave(signal_length, amp, sigma, dc):
    '''Get Gausssian P wave shape.'''
    data = scipy.signal.gaussian(signal_length, sigma) * amp + dc
    return data

def GetFittingCurve(
        len_sig,
        gpos,
        gsigma,
        gamp,
        hermit_coefs,
        ):
    '''Return fitting curve from parameters.'''
    fitting_curve = np.zeros(len_sig,)
    gaussian_segment = GetGaussianPwave(gsigma, gamp, gsigma / 6.0, 0)

    # Crop gaussian_segment
    left_cut_len = 0
    right_cut_len = 0
    if gpos < 0:
        left_cut_len = - gpos
    if gpos + gsigma > len_sig:
        right_cut_len = gpos + gsigma - len_sig

    gaussian_segment = gaussian_segment[left_cut_len:len(gaussian_segment) - right_cut_len]

    if gpos > 0:
        gaussian_curve = np.append(np.array([0.0, ] * (gpos - 1)),
                np.array(gaussian_segment))
    else:
        gaussian_curve = np.array(gaussian_segment)
    if gaussian_curve.size < len_sig:
        gaussian_curve = np.append(gaussian_curve,
                np.array([0.0,] * (len_sig - gaussian_curve.size)))

    for level, coef in zip(xrange(0,HermitFunction_max_level), hermit_coefs):
        fitting_curve += HermitFunction(level, int(len_sig)) * coef

    # print fitting_curve.shape
    # print 'gpos = ', gpos
    # print gaussian_curve.shape
    # pdb.set_trace()
    fitting_curve += gaussian_curve
    return fitting_curve

def MakeModel(sig_in, p_wave_length, fs = 250.0):
    '''Create P wave delineation model for MCMC.'''
    # Load ECG segment array
    raw_sig = sig_in
    raw_sig = np.array(raw_sig, dtype = np.float32)
    # Length of the ECG segment 
    len_sig = raw_sig.size


    hermit_coefs = list()
    # Dc baseline
    # coef = pymc.Normal('hc0', mu = 0, tau = 0.003)
    coef = DiscreteUniform('hc0', lower=-300,
            upper= 0, doc='hc0')
    hermit_coefs.append(coef)
    for ind in xrange(1, HermitFunction_max_level):
        coef = pymc.Normal('hc%d' % ind, mu = 0, tau = 0.003)
        hermit_coefs.append(coef)

    # upper_limit = min(50 * fs / 250.0, len_sig - 1)
    # lower_limit = 15 * fs / 250.0
    upper_limit = int(100 / 500.0 * fs)
    upper_limit = min(len_sig - 1, upper_limit)
    lower_limit = int(40 / 500.0 * fs)
    if lower_limit >= upper_limit:
        lower_limit = 1
    # print 'lower:', lower_limit
    # print 'upper:', upper_limit
    # print 'len_sig:', len_sig

    gaussian_sigma = DiscreteUniform('gaussian_sigma',
            lower = lower_limit,
            upper = upper_limit,
            doc='Gaussian wave width')

    # Max amp is around 200
    gaussian_amplitude = pymc.Normal('gaussian_amplitude', 0, 2e-3)
        
    # Baseline coefficients
    baseline_coefs = list()
    for ind in xrange(0, 5):
        baseline_coefs.append(pymc.Normal('bs%d' % ind, 0, 1.0))

    # P_dc = pymc.Normal('P_dc', 0, 1.0)
    # P_sigma = pymc.Normal('P_sigma', 7.0 * fs / 250.0, 0.2)
    # # P_pos = pymc.Normal('P_pos', len_sig - 35, 1.0)
    # P_pos = DiscreteUniform('P_pos', lower=45, upper=len_sig - 30, doc='WaveCetner[index]')
    # P_amp = pymc.Normal('P_amp', 1.0, 0.25)

    @pymc.stochastic(dtype=int)
    def gaussian_start_position(
            value = int(40 * fs / 500.0),
            gs = gaussian_sigma,
            ):
        ''' Start position of Gassian wave.'''
        left = int(40 * fs / 500.0)
        # right = len_sig - 1
        # right = len_sig - 1
        right = int(len_sig - 0.25 * gs)
        # print 'gs = ', gs
        # print 'right = ', right
        # print 'left = ', left

        # right = len_sig - gs
        # left = max(0, int(len(sig_in) * 2.0 / 3.0))
        if value < left or value > right:
            # Invalid values
            return -np.inf
        else:
            # gaussian log-prior
            center_index = int((left + right) * 0.9)
            # center_index = int(right - 1)
            d = int(max(center_index - left, right - center_index))
            M = 2 * d + 1
            pos_gaussian = scipy.signal.gaussian(M, 10 * fs / 500.0)
            pos_gaussian = pos_gaussian[M / 2 - center_index + int(left):M / 2 - center_index + int(right) + 1]
            p_sum = np.sum(pos_gaussian)
            pos_gaussian /= p_sum

            # debug
            # plt.figure(1)
            # plt.clf()
            # plt.plot(xrange(left, right + 1), pos_gaussian)
            # plt.grid(True)
            # plt.title('ECG')
            # plt.show()

            return np.log(pos_gaussian[int(value) - int(left)])

    @deterministic(plot=False)
    def wave_shell(
            gpos = gaussian_start_position,
            gs = gaussian_sigma,
            gamp = gaussian_amplitude,
            hermit_coefs = hermit_coefs,
            ):
        ''' Concatenate wave.'''

        fitting_curve = GetFittingCurve(len_sig,
                gpos, gs,
                gamp,
                hermit_coefs)
        return fitting_curve 



    ecg = pymc.Normal('ecg', mu=wave_shell, tau = 4e4, value=raw_sig, observed=True)

    return locals()
