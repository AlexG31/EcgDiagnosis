#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import json
import pymc
import numpy as np
from pymc import MCMC
import scipy.signal as signal
import pdb
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
from QTdata.loadQTdata import QTloader



def GetModel(sig_seg):
    '''Get Hermit Model Test.'''

    raw_sig = sig_seg
    # Normalize
    min_val = np.min(raw_sig)
    max_val = np.max(raw_sig)
    raw_sig -= min_val
    raw_sig /= (max_val - min_val)


    # Hermit functions
    def HermitFunction(level, size):
        '''Return hermit function.'''
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
        wave_width = 50
        x_ratio = 6.0 / wave_width
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


    @deterministic(plot=False)
    def wave_shell(hc0=hc0,
            hc1=hc1,
            hc2=hc2,
            hc3=hc3,
            hc4=hc4,
            hc5=hc5,
            ):
        ''' Concatenate wave.'''

        coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
        out = np.zeros(len_sig,)
        for level, coef in zip(xrange(0,6), coefs):
            out += HermitFunction(level, len_sig) * coef
        return out



    ecg = pymc.Normal('ecg', mu=wave_shell, tau = 4e4, value=raw_sig, observed=True)

    return locals()

qt = QTloader()
sig = qt.load('sel100')
raw_sig = sig['sig2']
sig_seg = raw_sig[100:140]
hermit_model = GetModel(sig_seg)
M = MCMC(hermit_model)

M.sample(iter = 1000, burn = 500, thin = 10)


hc0 = np.mean(M.trace('hc0')[:])
hc1 = np.mean(M.trace('hc1')[:])
hc2 = np.mean(M.trace('hc2')[:])
hc3 = np.mean(M.trace('hc3')[:])
hc4 = np.mean(M.trace('hc4')[:])
hc5 = np.mean(M.trace('hc5')[:])


# Fitting curve
len_sig = hermit_model['len_sig']
coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
out = np.zeros(len_sig,)
for level, coef in zip(xrange(0,6), coefs):
    out += hermit_model['HermitFunction'](level, len_sig) * coef

fitting_sig = out

plt.figure(2)
# Plot fitting curve
plt.plot(fitting_sig, label = 'fitting curve')
plt.plot(hermit_model['raw_sig'], label = 'ECG')
plt.title('ECG')
plt.legend()
plt.grid(True)
# plt.figure(1)
# plt.hist(xlist)
plt.show()
