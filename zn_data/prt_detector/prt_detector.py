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
from dpi.DPI_QRS_Detector import DPI_QRS_Detector
from dpi.QrsTypeDetector import QrsTypeDetector
from p_detection.PDelin_Gaussian import PDelineator

# t_delineation = importlib.import_module('t_detection.t-delineation')
t_delineation = importlib.import_module('t_detection.t-delineation')
# t_gaussian_model = importlib.import_module('t_detection.t-gaussian-model')
TDelineator = t_delineation.TDelineator
# import t_delineation.TDelineator as TDelineator

class PrtDetector(object):
    def __init__(self):
        pass
    def detect(self,raw_sig, fs):
        '''Run detection.
        This function assumes raw_signal's maximum possible voltage is
        more than 2000.
        '''
        # Where the detection results are saved
        result_dict = dict()

        print 'Start testing R with DPI algorithm.'
        debug_info = dict()
        debug_info['time_cost'] = True
        r_detector = DPI_QRS_Detector(debug_info = debug_info)
        r_list = r_detector.QRS_Detection(raw_sig, fs)

        filtered_sig = r_detector.HPF(raw_sig, fs = fs, fc = 2.0)
        filtered_sig = np.array(filtered_sig, dtype = np.float32)
        len_filtered_sig = len(filtered_sig)

        # Detect qrs type
        qrs_type = QrsTypeDetector(fs)
        # Detect T wave
        t_detector = TDelineator(None, None, None)
        # Detect P wave
        p_detector = PDelineator(None, fs = fs)

        for ind in xrange(0, len(r_list)):
            # Qrs analysis
            r_pos = r_list[ind]
            qrs_pos, qrs_text = qrs_type.GetQrsType(filtered_sig,
                    r_pos - 10, r_pos, r_pos + 10,
                    debug_plot = False)
            # Skipping qrs positions that exceeds boudary.
            if max(qrs_pos) >= len_filtered_sig:
                print 'Warning: qrs index >= length of signal!'
                continue

            # save to result
            if 'Ronset' not in result_dict:
                result_dict['Ronset'] = list()
            if 'Rpeak' not in result_dict:
                result_dict['Rpeak'] = list()
            if 'Roffset' not in result_dict:
                result_dict['Roffset'] = list()
            result_dict['Ronset'].append(qrs_pos[0])
            result_dict['Rpeak'].append(qrs_pos[1])
            result_dict['Roffset'].append(qrs_pos[2])


        qrs_pos_list = zip(result_dict['Ronset'], result_dict['Rpeak'],
                result_dict['Roffset'])
        # init
        result_dict['T_info'] = list()
        result_dict['P_info'] = list()

        for ind in xrange(0, len(qrs_pos_list) - 1):
        # for ind in xrange(0, 2):
            qrs_pos_current = qrs_pos_list[ind]
            qrs_pos_next = qrs_pos_list[ind + 1]

            # T detection
            t_result = t_detector.detect(filtered_sig, fs,
                    qrs_pos_current,
                    qrs_pos_next,
                    step = 4,
                    iter_count = 2000,
                    burn_count = 1000)
            if t_result is not None:
                result_dict['T_info'].append(t_result)

            # P detection
            p_result = p_detector.detect(filtered_sig, fs, qrs_pos_current,
                    qrs_pos_next,
                    step = 4,
                    iter_count = 2000,
                    burn_count = 1000,
                    )
            if p_result is not None:
                result_dict['P_info'].append(p_result)

        return result_dict



def plot(raw_sig, fs, results):
    '''Plot results.'''

    plt.figure(1)
    plt.clf()
    plt.plot(raw_sig, label='ECG')
    for label in ['Rpeak', 'Ronset', 'Roffset']:
        poslist = results[label]
        amplist = [raw_sig[x] for x in poslist]
        plt.plot(poslist, amplist, 'o', markersize = 12,
                alpha = 0.5, label = label)
    for label in ['Tpeak', 'Toffset', 'Tonset']:
        local_poslist = [x[label] for x in results['T_info']]
        global_bias_list = [x['global_bias'] for x in results['T_info']]
        poslist = [x[0] + x[1] for x in zip(local_poslist, global_bias_list)]
        amplist = [raw_sig[x] for x in poslist]
        plt.plot(poslist, amplist, 'o', markersize = 12,
                alpha = 0.5, label = label)

    # for segment_range in [x['segment_range'] for x in results['T_info']]:
        # poslist = xrange(segment_range[0], segment_range[1])
        # amplist = [raw_sig[x] for x in poslist]
        # plt.plot(poslist, amplist, lw = 6, alpha = 0.2, color = (1.0, 0.0, 0.4),
                # label = 'T segment')

    for label in ['Ppeak', 'Poffset', 'Ponset']:
        local_poslist = [x[label] for x in results['P_info']]
        global_bias_list = [x['global_bias'] for x in results['P_info']]
        poslist = [x[0] + x[1] for x in zip(local_poslist, global_bias_list)]
        amplist = [raw_sig[x] if x < len(raw_sig) else 0 for x in poslist]
        plt.plot(poslist, amplist, 'o', markersize = 12,
                alpha = 0.5, label = label)
    # for segment_range in [x['segment_range'] for x in results['P_info']]:
        # poslist = xrange(segment_range[0], segment_range[1])
        # amplist = [raw_sig[x] for x in poslist]
        # plt.plot(poslist, amplist, lw = 6, alpha = 0.2, color = (1.0, 0.0, 0.4),
                # label = 'P segment')
            

    # plt.xlim((results['Ronset'][0], results['Roffset'][4]))
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    with codecs.open('./diagnosis_info.json', 'r', 'utf8') as fin:
        dinfo = json.load(fin)

    prt_detector = PrtDetector()
    fs = 500.0
    zn_record_index = 0
    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            zn_record_index += 1
            print 'record index = ', zn_record_index
            print '.' * zn_record_index
            print 'Diagnosis:', diagnosis_text

            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name

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
                # test_on_zn(sig_data, record_ID, leadname, fs)
                raw_sig = sig_data['II'][:]
                results = prt_detector.detect(raw_sig, fs)
                plot(raw_sig, fs, results)
                break
