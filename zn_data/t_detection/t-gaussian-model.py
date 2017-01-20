#encoding:utf8

# MCMC Model
import os
import sys
import pymc
import pdb
import math
import scipy.signal
import matplotlib.pyplot as plt
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np
import json

# Hermit functions
HermitFunction_max_level = 8
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
        common_length,
        gpos,
        gsigma,
        gamp,
        hermit_coefs,
        ):
    '''Return fitting curve from parameters.'''
    fitting_curve = np.zeros(common_length,)
    gaussian_segment = GetGaussianPwave(gsigma, gamp, gsigma / 6.0, 0)

    # Crop gaussian_segment
    left_cut_len = 0
    right_cut_len = 0
    if gpos < 0:
        left_cut_len = - gpos
    if gpos + gsigma > common_length:
        right_cut_len = gpos + gsigma - common_length

    gaussian_segment = gaussian_segment[left_cut_len:len(gaussian_segment) - right_cut_len]

    if gpos > 0:
        gaussian_curve = np.append(np.array([0.0, ] * (gpos - 1)),
                np.array(gaussian_segment))
    else:
        gaussian_curve = np.array(gaussian_segment)
    if gaussian_curve.size < common_length:
        gaussian_curve = np.append(gaussian_curve,
                np.array([0.0,] * (common_length - gaussian_curve.size)))

    for level, coef in zip(xrange(0,HermitFunction_max_level), hermit_coefs):
        fitting_curve += HermitFunction(level, int(common_length)) * coef

    # print fitting_curve.shape
    # print 'gpos = ', gpos
    # print gaussian_curve.shape
    # pdb.set_trace()
    fitting_curve += gaussian_curve
    return fitting_curve

def MakeModel(sig1, fs, max_displacement = 10, ignore_edge_length = 10):
    '''Create P wave delineation model for MCMC.'''
    # Load ECG segment array
    sig1 = np.array(sig1, dtype = np.float32)

    # Length of the ECG segment 
    len_sig1 = sig1.size

    # Displacement variables
    # p1 = DiscreteUniform('p1', lower=0,
            # upper= max_displacement, doc='StartofSig1')
    # p2 = DiscreteUniform('p2', lower=0,
            # upper=max_displacement, doc='StartofSig2')

    common_length = len_sig1
    # common_length = DiscreteUniform('common_length',
            # lower = min(len_sig1 - p1.value, len_sig2 - p2) - max_displacement,
            # upper = min(len_sig1 - p1.value, len_sig2 - p2),
            # doc = 'CommonLength')
    
    hermit_coefs = list()
    # Dc baseline
    # coef = pymc.Normal('hc0', mu = 0, tau = 0.003)
    coef = DiscreteUniform('hc0', lower=-300,
            upper= 0, doc='hc0')
    hermit_coefs.append(coef)
    for ind in xrange(1, HermitFunction_max_level):
        coef = pymc.Normal('hc%d' % ind, mu = 0, tau = 0.003)
        hermit_coefs.append(coef)

    # wave_diff_sigma = pymc.Normal('wave_diff_sigma', 0, 1.0)
    # gaussian_sigma = pymc.HalfNormal('gaussian_sigma',
            # 0, (100.0 / 3.0 / fs) ** 2)
    upper_limit = min(50 * fs / 250.0, common_length - 1)
    lower_limit = 15 * fs / 250.0
    if lower_limit >= upper_limit:
        lower_limit = 1
    gaussian_sigma = DiscreteUniform('gaussian_sigma',
            lower=lower_limit,
            upper= upper_limit,
            doc='Gaussian wave width')

    # Max amp is around 200
    gaussian_amplitude = pymc.Normal('gaussian_amplitude', 0, 2e-3)
    # @pymc.stochastic(dtype=int)
    # def gaussian_amplitude(
            # value = 0,
            # ):
        # ''' Amplitude of Gassian wave.'''
        # if value < -300 or value > 300:
            # # Invalid values
            # return -np.inf
        # else:
            # # 2 gaussian log-prior
            # neg_gaussian = scipy.signal.gaussian(801, 120)
            # pos_gaussian = scipy.signal.gaussian(901, 110)
            # prior_gaussian = neg_gaussian[200:] + pos_gaussian[:601]
            
            # # Normalization
            # p_sum = np.sum(prior_gaussian)
            # prior_gaussian /= p_sum

            # return np.log(prior_gaussian[int(value)])


    # @pymc.stochastic(dtype=int)
    # def gaussian_start_position(
            # value = 0,
            # gs = gaussian_sigma,
            # ):
        # ''' Start position of Gassian wave.'''
        # if value < -0.35 * gs or value > common_length - 0.65 * gs:
            # # Invalid values
            # return -np.inf
        # else:
            # # Uniform log-likelihood
            # return -np.log(common_length - gs + 1)

    @pymc.stochastic(dtype=int)
    def gaussian_start_position(
            value = 0,
            gs = gaussian_sigma,
            ):
        ''' Start position of Gassian wave.'''
        left = -0.35 * gs
        right = common_length - 0.65 * gs
        if value < left or value > right:
            # Invalid values
            return -np.inf
        else:
            # gaussian log-prior
            d = int(max(40 - left, right - 40))
            M = 2 * d + 3
            pos_gaussian = scipy.signal.gaussian(M, 3)
            pos_gaussian = pos_gaussian[M / 2 - 40 + int(left):M / 2 - 40 + int(right) + 1]
            p_sum = np.sum(pos_gaussian)
            pos_gaussian /= p_sum

            return np.log(pos_gaussian[int(value) - int(left)])
    
    @deterministic(plot=False)
    def wave_diff(
            gpos = gaussian_start_position,
            gs = gaussian_sigma,
            gamp = gaussian_amplitude,
            hermit_coefs = hermit_coefs,
            ):
        ''' Concatenate wave.'''

        out = sig1[:common_length]
        fitting_curve = GetFittingCurve(common_length,
                gpos, gs,
                gamp,
                hermit_coefs)
        # fitting_curve = np.zeros(common_length,)
        # gaussian_segment = GetGaussianPwave(6 * gs, gamp, gs, gbase)
        # if gpos > 0:
            # gaussian_curve = np.append(np.array([0.0, ] * (gpos - 1)),
                    # np.array(gaussian_segment))
        # else:
            # gaussian_curve = np.array(gaussian_segment)
        # if gaussian_curve.size < common_length:
            # gaussian_curve = np.append(gaussian_curve, np.array([0.0,] * (common_length - gaussian_curve.size)))

        # for level, coef in zip(xrange(0,8), hermit_coefs):
            # fitting_curve += HermitFunction(level, int(comp_length)) * coef
        # fitting_curve += gaussian_curve

        
        return out - fitting_curve


    # @deterministic(plot=False)
    # def wave_shape_sigma(
            # sigma1 = wave_diff_sigma,
            # ):
        # ''' Difference sigma because of wave difference.'''

        # out = abs(sigma1 + 1.0)
        # return out

    diff_sig = pymc.Normal('diff_sig', mu=wave_diff,
            tau = 0.1,
            value = [0,] * common_length,
            observed=True)

    return locals()
