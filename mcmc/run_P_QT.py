#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
from pymc.Matplot import plot
import json
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

# plt.plot(raw_sig, label = 'lead2')
# plt.plot(sig['sig'], label = 'lead1')
# plt.title('%s' % reclist[rec_ind])
# plt.show()

fs = 250.0

r_detector = DPI_QRS_Detector()
r_list = r_detector.QRS_Detection(raw_sig)
print 'R detection time cost : %d' % (time.time() - start_time)

amp_list = [raw_sig[x] for x in r_list]
# index of R wave
ind = 371
region_left = r_list[ind]
region_right = r_list[ind + 1]
QR_length = fs / 20.0
PR_length = 0.65 * (region_right - region_left)

cut_x_list = [int(region_right - PR_length), int(region_right - QR_length)]
sin_list = [math.sin(x / fs * math.pi) * 30 for x in xrange(0, region_right - region_left)]
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
# with open('./tmp_signal.json', 'w') as fout:
    # json.dump(sig_seg, fout, indent = 4)

# pdb.set_trace()


p_wave_length = 38.0 * fs / 250.0
p_model = P_model.MakeModel(sig_seg, p_wave_length, fs = fs)
M = MCMC(p_model)

M.sample(iter = 3000, burn = 1500, thin = 8)


hc0 = np.mean(M.trace('hc0')[:])
hc1 = np.mean(M.trace('hc1')[:])
hc2 = np.mean(M.trace('hc2')[:])
hc3 = np.mean(M.trace('hc3')[:])
hc4 = np.mean(M.trace('hc4')[:])
hc5 = np.mean(M.trace('hc5')[:])
amp = np.mean(M.trace('P_amp')[:])
pos = np.mean(M.trace('P_pos')[:])

# plot(M)
# plt.show()
# pdb.set_trace()


# Fitting curve
len_sig = p_model['len_sig']
coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
# out = np.zeros(len_sig,)
# for level, coef in zip(xrange(0,6), coefs):
    # out += P_model.HermitFunction(level, len_sig) * coef


p_shape_coefs = [hc0, hc1, hc2, hc3, hc4, hc5,]
p_wave_list = np.zeros(p_wave_length,)
for level, coef in zip(xrange(0,6), p_shape_coefs):
    p_wave_list += P_model.HermitFunction(level, p_wave_length) * coef * amp


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
fitting_sig = np.copy(baseline_curve)
for p_ind, p_val in enumerate(p_wave_list):
    sig_ind = p_ind - p_wave_length / 2.0 + pos
    sig_ind = int(sig_ind)
    # Out of scope
    if sig_ind >= len_sig or sig_ind < 0:
        continue
    fitting_sig[sig_ind] += p_val
    part_p_wave.append(fitting_sig[sig_ind])
    part_p_indexes.append(p_ind)

peak_bias = part_p_indexes[np.argmax(part_p_wave)]
peak_pos = int(pos + peak_bias - len(p_wave_list) / 2.0)


# ==================================================
#       Visualization of P x MCMC
# ==================================================
whole_sig_left = max(cut_x_list[0] - 50, 0)
whole_sig_bias = cut_x_list[0] - whole_sig_left
whole_sig_right = min(cut_x_list[1] + 50, len(raw_sig))
whole_sig = raw_sig[whole_sig_left:whole_sig_right]
# Lead2
whole_sig2 = sig['sig'][whole_sig_left:whole_sig_right]
if p_model['max_val'] - p_model['min_val'] > 1e-6:
    whole_sig = np.array(whole_sig)
    whole_sig -= p_model['min_val']
    whole_sig /= (p_model['max_val'] - p_model['min_val'])
    # Lead2
    whole_sig2 = np.array(whole_sig2)
    whole_sig2 -= p_model['min_val']
    whole_sig2 /= (p_model['max_val'] - p_model['min_val'])

plt.figure(1)
# Plot fitting curve
plt.plot(np.array(part_p_wave) * (np.max(whole_sig) - np.min(whole_sig)), label = 'Fitting curve')
plt.plot(whole_sig, label = 'ECG')
plt.plot(whole_sig2, linestyle = '--', label = 'Lead2')
plt.plot(peak_pos + whole_sig_bias, p_model['raw_sig'][peak_pos],
        'ro', markersize = 12,label = 'P pos')
plt.plot(xrange(whole_sig_bias, whole_sig_bias + len(fitting_sig)),
        fitting_sig, linewidth = 2, color = 'orange', label = 'fitting signal')
plt.plot(xrange(whole_sig_bias, whole_sig_bias + len(baseline_curve)),
        baseline_curve, linewidth = 3, color = 'black', alpha = 0.3,
        label = 'baseline')
plt.title('ECG %s' % reclist[rec_ind])
plt.legend()
plt.grid(True)
# plt.figure(1)
# plt.hist(xlist)
plt.show()
