#encoding:utf8
import importlib
import os
import codecs
import json
import glob
import sys
import scipy.io as sio
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

def test_on_zn(sig_struct, record_ID, leadname, fs, normalize_amplitude = False):
    '''Run T detection(T_onset, T_peak, T_offset) on 中关村数据(both leads)'''
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

    recname = record_ID
    # if os.path.exists('./results/zn/t_%s_%s.json' % (recname, leadname)):
        # return None

    result_dict['recname'] = recname
    result_dict['leadname'] = leadname
    result_dict['Ronset'] = list()
    result_dict['Tonset'] = list()
    result_dict['Roffset'] = list()
    result_dict['Toffset'] = list()
    result_dict['Tpeak'] = list()
    result_dict['Rpeak'] = list()
    result_dict['Tamplitude'] = list()
    result_dict['step'] = list()
    result_dict['T_global_bias'] = list()

    sig = sig_struct
    raw_sig = sig[leadname]

    from dpi.DPI_QRS_Detector import DPI_QRS_Detector
    from dpi.QrsTypeDetector import QrsTypeDetector

    # Detect qrs type
    qrs_type = QrsTypeDetector(fs)

    print 'Start testing R waves with DPI.'
    debug_info = dict()
    debug_info['time_cost'] = True
    r_detector = DPI_QRS_Detector(debug_info = debug_info)
    r_list = r_detector.QRS_Detection(raw_sig, fs)

    filtered_sig = r_detector.HPF(raw_sig, fs = fs, fc = 2.0)
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

        step = 4
        if qrs_pos[2] + 20 >= len(raw_sig) or r_list[ind + 1] >= len(raw_sig):
            print 'Warning: qrs_pos exceeds boundary!'
            continue
        sig1 = raw_sig[qrs_pos[2] + 20:r_list[ind + 1] - step * 20][0::step]
        remain_range = remove_edge_QRS(sig1)
        cur_t_global_bias = qrs_pos[2] + 20 + remain_range[0]

        # Skip short signals
        if remain_range[1] - remain_range[0] <= 20:
            continue

        sig1 = sig1[remain_range[0]:remain_range[1]]

        # HPF ECG segment
        len_sig1 = len(sig1)
        sig1 = r_detector.HPF(sig1, fs = fs, fc = 2.0)
        sig1 = sig1[0:len_sig1]

        # Convert to np array
        sig1 = np.array(sig1, dtype = np.float32)
        if np.max(sig1) - np.min(sig1) < 1e-6:
            raise Exception('sig1 is DC signal!')

        if normalize_amplitude:
            sig1 /= (np.max(sig1) - np.min(sig1))
            sig1 *= 2500.0
            # sig1 -= np.max(sig1)

        wcomp = TDelineator(sig1, float(fs) / step, t_gaussian_model)
        pos = wcomp.detection(iter_count = 5000, burn_count = 2500)
        #debug
        wcomp.trace_plot()
        wcomp.plot()

        T_poslist = wcomp.GetTpos()
        # with complement
        T_poslist = [x * step + cur_t_global_bias for x in T_poslist]

        result_dict['Ronset'].append(qrs_pos[0])
        result_dict['Rpeak'].append(qrs_pos[1])
        result_dict['Roffset'].append(qrs_pos[2])
        result_dict['Tonset'].append(T_poslist[0])
        result_dict['Tpeak'].append(T_poslist[1])
        result_dict['Toffset'].append(T_poslist[2])
        result_dict['Tamplitude'].append(wcomp.GetTamp())
        result_dict['step'].append(step)
        result_dict['T_global_bias'].append(cur_t_global_bias)


if __name__ == '__main__':
    with codecs.open('./diagnosis_info.json', 'r', 'utf8') as fin:
        dinfo = json.load(fin)

    # target_record_ID = 'MEDEXS120160203085531515'
    # target_record_ID = 'xdtscx20151116142148100'
    target_record_ID = 'MEDEXS120160216095333561'

    fs = 500.0
    zn_record_index = 0
    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            zn_record_index += 1
            if zn_record_index % 100 == 0:
                print 'record index = ', zn_record_index
                print '.' * int(zn_record_index / 1000)

            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name

            # Skipping
            if record_ID != target_record_ID:
                continue

            # Load signal mat file
            mat_file_list = glob.glob(os.path.join(
                current_folder, record_ID + '*.mat'))
            if len(mat_file_list) == 0:
                raise Exception('Signal mat file does not exist!')
            sig_data = sio.loadmat(mat_file_list[0])
            for key in sig_data.keys():
                sig_data[key] = np.squeeze(sig_data[key])

            for leadname in ['I', 'II', 'III', 'aVR', 'aVF', 'aVL',
                    'V1', 'V2', 'V3', 'V4','V5', 'V6']:
                if leadname != 'II':
                    continue
                print 'testing:', leadname
                test_on_zn(sig_data, record_ID, leadname, fs)



