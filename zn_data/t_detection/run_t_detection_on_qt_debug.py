#encoding:utf8
import importlib
import os
import json
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pymc
from pymc import MCMC
import pdb
import importlib

t_delineation = importlib.import_module('t-delineation')
t_gaussian_model = importlib.import_module('t-gaussian-model')
TDelineator = t_delineation.TDelineator
# import t_delineation.TDelineator as TDelineator

def test_on_qt(rec_ind, leadname):
    '''Run T detection(T_onset, T_peak, T_offset) on QTdb(both leads)'''
    def remove_edge_QRS(sig_in):
        left = 0
        right = len(sig_in)
        sig_out = sig_in
        while left < right:
            max_val = max(sig_out[left:right])
            if sig_out[left] == max_val:
                left += 1
            elif left + 1< right and sig_out[right - 1] == max_val:
                right -= 1
            else:
                break
        return (left, right)

    # QRS & T results
    result_dict = dict()

    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    reclist = qt.getreclist()
    recname = reclist[rec_ind]

    result_dict['recname'] = recname
    result_dict['leadname'] = leadname
    result_dict['Ronset'] = list()
    result_dict['Tonset'] = list()
    result_dict['Roffset'] = list()
    result_dict['Toffset'] = list()
    result_dict['Tpeak'] = list()
    result_dict['Rpeak'] = list()
    result_dict['Tamplitude'] = list()

    sig = qt.load(recname)
    raw_sig = sig[leadname]

    from dpi.DPI_QRS_Detector import DPI_QRS_Detector
    from dpi.QrsTypeDetector import QrsTypeDetector

    # Detect qrs type
    qrs_type = QrsTypeDetector(250.0)

    print 'Start testing R waves with DPI.'
    debug_info = dict()
    debug_info['time_cost'] = True
    r_detector = DPI_QRS_Detector(debug_info = debug_info)
    r_list = r_detector.QRS_Detection(raw_sig, 250.0)

    filtered_sig = r_detector.HPF(raw_sig, fs = 250.0, fc = 2.0)
    filtered_sig = np.array(filtered_sig, dtype = np.float32)
    filtered_sig *= 1000

    for ind in xrange(0, len(r_list) - 1):
        # Qrs analysis
        r_pos = r_list[ind]
        qrs_pos, qrs_text = qrs_type.GetQrsType(filtered_sig,
                r_pos - 10, r_pos, r_pos + 10,
                debug_plot = False)

        # save to result

        print qrs_text
        print qrs_pos

        step = 2
        if qrs_pos[2] + 20 >= len(raw_sig) or r_list[ind + 1] >= len(raw_sig):
            print 'Warning: qrs_pos exceeds boundary!'
            continue
        sig1 = raw_sig[qrs_pos[2] + 20:r_list[ind + 1]][0::step]
        remain_range = remove_edge_QRS(sig1)

        # Skip short signals
        if remain_range[1] - remain_range[0] <= 20:
            continue

        sig1 = sig1[remain_range[0]:remain_range[1]]

        # HPF ECG segment
        len_sig1 = len(sig1)
        sig1 = r_detector.HPF(sig1, fs = 250.0, fc = 2.0)
        sig1 = sig1[0:len_sig1]

        # Convert to np array
        sig1 = np.array(sig1, dtype = np.float32)
        if np.max(sig1) - np.min(sig1) < 1e-6:
            raise Exception('sig1 is DC signal!')

        sig1 /= (np.max(sig1) - np.min(sig1))
        sig1 *= 2500.0
        # sig1 -= np.max(sig1)

        wcomp = TDelineator(sig1, 250.0 / step, t_gaussian_model)
        pos = wcomp.detection(iter_count = 1000, burn_count = 500)

        T_poslist = wcomp.GetTpos()
        # debug
        wcomp.plot()

        result_dict['Ronset'].append(qrs_pos[0])
        result_dict['Rpeak'].append(qrs_pos[1])
        result_dict['Roffset'].append(qrs_pos[2])
        result_dict['Tonset'].append(T_poslist[0])
        result_dict['Tpeak'].append(T_poslist[1])
        result_dict['Toffset'].append(T_poslist[2])
        result_dict['Tamplitude'].append(wcomp.GetTamp())

        plt.figure(1)
        plt.clf()
        plt.plot(raw_sig, label = 'ECG')

        for c_label in ['Ronset', 'Rpeak', 'Roffset']:
            poslist = result_dict[c_label]
            amplist = [raw_sig[int(x)] for x in poslist]
            plt.plot(poslist, amplist, 'o', markersize = 12, label = c_label)

        T_poslist = [int(x * step + qrs_pos[2] + 20 + remain_range[0]) for x in T_poslist]
        amplist = [raw_sig[x] for x in T_poslist]
        plt.plot(T_poslist, amplist, '^', markersize = 12, label = 'T')

        plt.grid(True)
        plt.legend()
        plt.title(recname)
        # plt.xlim((qrs_pos[0] - 40, T_poslist[2] + 40))
        plt.show(block = False)
        pdb.set_trace()


if __name__ == '__main__':
    for ind in xrange(10, 105):
        print 'record index = ', ind
        print '.' * ind
        test_on_qt(ind, 'sig')
        test_on_qt(ind, 'sig2')
