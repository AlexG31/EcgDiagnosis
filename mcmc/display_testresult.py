#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import json
import pickle
import time
import P_model
import numpy as np
import math
from pymc import MCMC
import scipy.signal as signal
import pdb
from QTdata.loadQTdata import QTloader
from dpi.DPI_QRS_Detector import DPI_QRS_Detector

# Time statistics
start_time = time.time()
# write to json
qt = QTloader()
reclist = qt.getreclist()
rec_ind = 21
sig = qt.load(reclist[rec_ind])
raw_sig = sig['sig']

# Plot 2 lead signal
# plt.plot(raw_sig, label = 'lead2')
# plt.plot(sig['sig'], label = 'lead1')
# plt.title('%s' % reclist[rec_ind])
# plt.show()

fs = 250.0



# Write to json
with open('./tmp.pkl', 'r') as fin:
    result_dict = pickle.load(fin)

# ==================================================
#       Visualization of P x MCMC
# ==================================================
for ind in xrange(0, len(result_dict['bs0'])):
    cut_x_list = result_dict['segment_range'][ind]
    max_val = result_dict['max_val'][ind]
    min_val = result_dict['min_val'][ind]
    peak_global_pos = result_dict['peak_global_pos'][ind]
    bs0 = result_dict['bs0'][ind]
    bs1 = result_dict['bs1'][ind]
    bs2 = result_dict['bs2'][ind]
    bs3 = result_dict['bs3'][ind]
    bs4 = result_dict['bs4'][ind]
    hc0 = result_dict['hc0'][ind]
    hc1 = result_dict['hc1'][ind]
    hc2 = result_dict['hc2'][ind]
    hc3 = result_dict['hc3'][ind]
    hc4 = result_dict['hc4'][ind]
    hc5 = result_dict['hc5'][ind]
    P_amp = result_dict['P_amp'][ind]
    P_pos = result_dict['P_pos'][ind]

    peak_pos = peak_global_pos - cut_x_list[0]
    baseline_kx = [bs0, bs1,bs2,bs3, bs4,]
    p_shape_coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]

    whole_sig_left = max(cut_x_list[0] - 50, 0)
    whole_sig_bias = cut_x_list[0] - whole_sig_left
    whole_sig_right = min(cut_x_list[1] + 50, len(raw_sig))
    whole_sig = raw_sig[whole_sig_left:whole_sig_right]
    # Lead2
    whole_sig2 = sig['sig'][whole_sig_left:whole_sig_right]
    if max_val - min_val > 1e-6:
        whole_sig = np.array(whole_sig)
        whole_sig -= min_val
        whole_sig /= (max_val - min_val)
        # Lead2
        whole_sig2 = np.array(whole_sig2)
        whole_sig2 -= min_val
        whole_sig2 /= (max_val - min_val)

    # Fitting curve
    len_sig = cut_x_list[1] - cut_x_list[0]
    # P wave shape
    p_wave_length = 50.0 * fs / 250.0
    p_wave_list = np.zeros(p_wave_length,)
    for level, coef in zip(xrange(0,6), p_shape_coefs):
        p_wave_list += P_model.HermitFunction(level, p_wave_length) * coef * P_amp
    fitting_sig = np.array(p_wave_list)
    # Baseline shape
    baseline_curve = np.zeros(len_sig,)
    baseline_mat = P_model.GetBaselineMatrix(len_sig, fs)
    part_p_wave = list()
    part_p_indexes = list()
    for baseline_list, kx in zip(baseline_mat, baseline_kx):
        baseline_curve += baseline_list * kx
    for p_ind, p_val in enumerate(p_wave_list):
        sig_ind = p_ind - p_wave_length / 2.0 + P_pos
        sig_ind = int(sig_ind)
        # Out of scope
        if sig_ind >= len_sig or sig_ind < 0:
            continue
        baseline_curve[sig_ind] += p_val
        part_p_wave.append(baseline_curve[sig_ind])
        part_p_indexes.append(p_ind)
    # Plot fitting curve
    plt.figure(1)
    plt.plot(whole_sig, label = 'ECG')
    plt.plot(whole_sig2, label = 'Lead2')
    plt.plot(peak_pos + whole_sig_bias, whole_sig[peak_pos + whole_sig_bias],
            'ro', markersize = 12,label = 'P pos')
    plt.plot(xrange(whole_sig_bias, whole_sig_bias + len(baseline_curve)),
            baseline_curve, label = 'baseline')
    plt.title('ECG %s (Peak %d)' % (reclist[rec_ind], ind))
    plt.legend()
    plt.grid(True)
    # plt.figure(1)
    # plt.hist(xlist)
    plt.show()

