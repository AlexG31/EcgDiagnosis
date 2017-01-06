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

r_detector = DPI_QRS_Detector()
r_list = r_detector.QRS_Detection(raw_sig)
print 'R detection time cost : %d seconds.' % (time.time() - start_time)

# index of R wave
ind = 8
result_dict = dict()
# init
result_dict['hc0'] = list()
result_dict['hc1'] = list()
result_dict['hc2'] = list()
result_dict['hc3'] = list()
result_dict['hc4'] = list()
result_dict['hc5'] = list()
result_dict['P_amp'] = list()
result_dict['P_pos'] = list()
result_dict['bs0'] = list()
result_dict['bs1'] = list()
result_dict['bs2'] = list()
result_dict['bs3'] = list()
result_dict['bs4'] = list()
result_dict['min_val'] = list()
result_dict['max_val'] = list()
result_dict['peak_global_pos'] = list()
result_dict['segment_range'] = list()

for ind in xrange(0, 10):
    region_left = r_list[ind]
    region_right = r_list[ind + 1]
    QR_length = fs / 20.0
    PR_length = 0.65 * (region_right - region_left)

    cut_x_list = [int(region_right - PR_length), int(region_right - QR_length)]
    sin_list = [math.sin(x / fs * math.pi) * 30 for x in xrange(0, region_right - region_left)]
    # amp_list = [raw_sig[x] for x in r_list]
    # plt.plot(raw_sig)
    # plt.plot(xrange(region_left, region_right), sin_list)
    # plt.plot(r_list, amp_list, 'ro')
    # plt.plot(cut_x_list,
        # [raw_sig[x] for x in cut_x_list], 'rs', markersize = 12)
    # plt.title('ECG %s' % reclist[rec_ind])
    # plt.grid(True)
    # # plt.xlim((r_list[17],r_list[18]))
    # plt.show()

    sig_seg = raw_sig[cut_x_list[0]:cut_x_list[1]]

    # plt.figure(2)
    # plt.plot(sig_seg, label = 'sig segment')
    # plt.title('Signal Segment')


    p_wave_length = 38.0 * fs / 250.0
    p_model = P_model.MakeModel(sig_seg, p_wave_length, fs = fs)
    M = MCMC(p_model)

    M.sample(iter = 3000, burn = 1500, thin = 10)


    hc0 = np.mean(M.trace('hc0')[:])
    hc1 = np.mean(M.trace('hc1')[:])
    hc2 = np.mean(M.trace('hc2')[:])
    hc3 = np.mean(M.trace('hc3')[:])
    hc4 = np.mean(M.trace('hc4')[:])
    hc5 = np.mean(M.trace('hc5')[:])
    amp = np.mean(M.trace('P_amp')[:])
    pos = np.mean(M.trace('P_pos')[:])


    # Fitting curve
    len_sig = p_model['len_sig']
    p_shape_coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]

    p_wave_list = np.zeros(p_wave_length,)
    for level, coef in zip(xrange(0,6), p_shape_coefs):
        p_wave_list += P_model.HermitFunction(level, p_wave_length) * coef * amp
    fitting_sig = np.array(p_wave_list)


    bs0 = np.mean(M.trace('bs0')[:])
    bs1 = np.mean(M.trace('bs1')[:])
    bs2 = np.mean(M.trace('bs2')[:])
    bs3 = np.mean(M.trace('bs3')[:])
    bs4 = np.mean(M.trace('bs4')[:])
    baseline_kx = [bs0, bs1,bs2,bs3, bs4,]

    baseline_curve = np.zeros(len_sig,)
    # add baseline
    baseline_mat = P_model.GetBaselineMatrix(len_sig, fs)
    part_p_wave = list()
    part_p_indexes = list()
    for baseline_list, kx in zip(baseline_mat, baseline_kx):
        baseline_curve += baseline_list * kx
    for p_ind, p_val in enumerate(p_wave_list):
        sig_ind = p_ind - p_wave_length / 2.0 + pos
        sig_ind = int(sig_ind)
        # Out of scope
        if sig_ind >= len_sig or sig_ind < 0:
            continue
        baseline_curve[sig_ind] += p_val
        part_p_wave.append(baseline_curve[sig_ind])
        part_p_indexes.append(p_ind)

    peak_bias = part_p_indexes[np.argmax(part_p_wave)]
    peak_pos = int(pos + peak_bias - len(fitting_sig) / 2.0)
    peak_global_pos = peak_pos + cut_x_list[0]


    # Save to result dict
    result_dict['hc0'].append(hc0)
    result_dict['hc1'].append(hc1)
    result_dict['hc2'].append(hc2)
    result_dict['hc3'].append(hc3)
    result_dict['hc4'].append(hc4)
    result_dict['hc5'].append(hc5)
    result_dict['P_amp'].append(amp)
    result_dict['P_pos'].append(pos)
    result_dict['bs0'].append(bs0)
    result_dict['bs1'].append(bs1)
    result_dict['bs2'].append(bs2)
    result_dict['bs3'].append(bs3)
    result_dict['bs4'].append(bs4)
    result_dict['min_val'].append(p_model['min_val'])
    result_dict['max_val'].append(p_model['max_val'])
    result_dict['segment_range'].append(cut_x_list)
    result_dict['peak_global_pos'].append(peak_global_pos)

    continue



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

# ====Write to json====
# For display_results.py
with open('./tmp.pkl', 'w') as fout:
    pickle.dump(result_dict, fout)
