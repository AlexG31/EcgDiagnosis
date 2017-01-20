#encoding:utf8
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import json
import pickle
import time
import P_model_Gaussian
import numpy as np
import math
from pymc import MCMC
import scipy.signal as signal
import pdb
from QTdata.loadQTdata import QTloader
from dpi.DPI_QRS_Detector import DPI_QRS_Detector



zn_font = matplotlib.font_manager.FontProperties(fname = '/usr/share/fonts/truetype/simsun.ttc')

class PDelineator(object):
    def __init__(self, raw_sig, fs = 250.0, p_wave_lengthMs = 38 * 4):
        '''Delineator of P wave.'''
        self.raw_sig = raw_sig
        self.fs = fs
        self.p_wave_length = p_wave_lengthMs * fs / 1000.0


        self.r_detector = DPI_QRS_Detector()
        if raw_sig is not None:
            start_time = time.time()
            self.r_list = self.r_detector.QRS_Detection(raw_sig)
            print 'R detection time cost : %d seconds.' % (time.time() - start_time)
            self.raw_sig = self.r_detector.HPF(raw_sig, fs = fs, fc = 3.0)

        # index of R wave
        result_dict = dict()
        # init
        result_dict['gamp'] = list()
        result_dict['gsigma'] = list()
        result_dict['gpos'] = list()
        result_dict['hermit_coefs'] = list()
        result_dict['segment_range'] = list()
        result_dict['peak_global_pos'] = list()

        self.result_dict = result_dict

    def detect(self,
            filtered_sig,
            fs,
            qrs_pos_current,
            qrs_pos_next,
            step = 2,
            iter_count = 1000,
            burn_count = 500,
            ):
        '''Detect and returns P wave detection results.
            Note:
                This function returns None when detection fails.
        '''
        # r_list = self.r_list
        results = dict()
        raw_sig = filtered_sig
        p_wave_length = self.p_wave_length
        r_detector = self.r_detector

        # region_left = r_list[ind]
        # region_right = r_list[ind + 1]
        region_left = qrs_pos_current[2]
        region_right = qrs_pos_next[0]
        # QR_length = fs / 46.0
        # PR_length = 0.5 * (region_right - region_left)
        PQ_length = 0.5 * (region_right - region_left)

        # segment_range = [int(region_right - PR_length), int(region_right - QR_length)]
        segment_range = [int(region_right - PQ_length), int(region_right)]
        if segment_range[1] - segment_range[0] <= 0:
            return None

        sig_seg = raw_sig[segment_range[0]:segment_range[1]][::step]
        sig_seg = r_detector.HPF(sig_seg, fs = fs, fc = 3.0)
        if len(sig_seg) <= 75:
            print 'R-R interval %d is too short!' % len(sig_seg)
            return None


        p_model = P_model_Gaussian.MakeModel(sig_seg, p_wave_length, fs = fs / step)
        M = MCMC(p_model)

        M.sample(iter = iter_count, burn = burn_count, thin = 10)

        # retrieve parameters
        # hermit_coefs = list()
        # for h_ind in xrange(0, P_model_Gaussian.HermitFunction_max_level):
            # hermit_value = np.mean(M.trace('hc%d' % h_ind)[:])
            # hermit_coefs.append(hermit_value)
        # P wave shape parameters
        gpos = np.mean(M.trace('gaussian_start_position')[:])
        gsigma = np.mean(M.trace('gaussian_sigma')[:])
        gamp = np.mean(M.trace('gaussian_amplitude')[:])

        # print 'Results:'
        # print 'gpos = ', gpos
        # print 'gsigma = ', gsigma
        # print 'gamp = ', gamp
        
        # len_sig = segment_range[1] - segment_range[0]
        # fitting_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                # gpos, gsigma,
                # gamp,
                # hermit_coefs)
        # baseline_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                # gpos, gsigma,
                # 0,
                # hermit_coefs)
        # plt.figure(1)
        # plt.clf()
        # plt.plot(sig_seg, label = 'ECG')
        # plt.plot(fitting_curve, label = 'fitting curve')
        # plt.plot(baseline_curve,
                # linestyle = '--',
                # alpha = 0.3, lw = 2,
                # label = 'baseline curve')
        # plt.plot(gpos, fitting_curve[gpos], 'r^', markersize = 12)
        # plt.legend()
        # plt.grid(True)
        # if 'plot_title' in debug_info:
            # plt.title(debug_info['plot_title'], fontproperties = zn_font)
        # if 'plot_xlabel' in debug_info:
            # plt.xlabel(debug_info['plot_xlabel'])
        # plt.show(block = False)
        # plt.savefig('./results/tmp/%d.png' % int(time.time()))


        # peak_pos = int(gpos + gsigma / 2.0)
        # peak_global_pos = peak_pos + segment_range[0]


        # Save to result dict
        gpos *= step
        gsigma *= step

        results['Ponset'] = int(gpos)
        results['Ppeak'] = int(gpos + gsigma / 2)
        results['Poffset'] = int(gpos + gsigma)
        results['global_bias'] = segment_range[0]
        results['amplitude'] = gamp
        results['step'] = step
        results['segment_range'] = segment_range

        return results

    def run(self, debug_info = dict()):
        '''Run delineation process for each R-R interval.'''

        r_list = self.r_list
        result_dict = self.result_dict
        fs = self.fs
        raw_sig = self.raw_sig
        p_wave_length = self.p_wave_length
        r_detector = DPI_QRS_Detector()

        for ind in xrange(0, len(r_list) - 1):
            print 'Progress: %d R-R intervals left.' % (len(r_list) - 1 - ind)
            if ind > 1:
                print 'Debug break.'
                break
            region_left = r_list[ind]
            region_right = r_list[ind + 1]
            QR_length = fs / 46.0
            PR_length = 0.5 * (region_right - region_left)

            cut_x_list = [int(region_right - PR_length), int(region_right - QR_length)]

            sig_seg = raw_sig[cut_x_list[0]:cut_x_list[1]]
            sig_seg = r_detector.HPF(sig_seg, fs = fs, fc = 3.0)
            sig_seg = sig_seg[:cut_x_list[1] - cut_x_list[0]]
            if len(sig_seg) <= 75:
                print 'R-R interval %d is too short!' % len(sig_seg)
                continue


            p_model = P_model_Gaussian.MakeModel(sig_seg, p_wave_length, fs = fs)
            M = MCMC(p_model)

            M.sample(iter = 2000, burn = 1000, thin = 10)

            # retrieve parameters
            hermit_coefs = list()
            for h_ind in xrange(0, P_model_Gaussian.HermitFunction_max_level):
                hermit_value = np.mean(M.trace('hc%d' % h_ind)[:])
                hermit_coefs.append(hermit_value)
            # P wave shape parameters
            gpos = np.mean(M.trace('gaussian_start_position')[:])
            gsigma = np.mean(M.trace('gaussian_sigma')[:])
            gamp = np.mean(M.trace('gaussian_amplitude')[:])

            print 'Results:'
            print 'gpos = ', gpos
            print 'gsigma = ', gsigma
            print 'gamp = ', gamp
            
            len_sig = cut_x_list[1] - cut_x_list[0]
            fitting_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    gamp,
                    hermit_coefs)
            baseline_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    0,
                    hermit_coefs)
            plt.figure(1)
            plt.clf()
            plt.plot(sig_seg, label = 'ECG')
            plt.plot(fitting_curve, label = 'fitting curve')
            plt.plot(baseline_curve,
                    linestyle = '--',
                    alpha = 0.3, lw = 2,
                    label = 'baseline curve')
            plt.plot(gpos, fitting_curve[gpos], 'r^', markersize = 12)
            plt.legend()
            plt.grid(True)
            if 'plot_title' in debug_info:
                plt.title(debug_info['plot_title'], fontproperties = zn_font)
            if 'plot_xlabel' in debug_info:
                plt.xlabel(debug_info['plot_xlabel'])
            plt.show(block = False)
            plt.savefig('./results/tmp/%d.png' % int(time.time()))


            peak_pos = int(gpos + gsigma / 2.0)
            peak_global_pos = peak_pos + cut_x_list[0]


            # Save to result dict
            result_dict['gamp'].append(gamp)
            result_dict['gsigma'].append(gsigma)
            result_dict['gpos'].append(gpos)
            result_dict['hermit_coefs'].append(hermit_coefs)
            result_dict['segment_range'].append(cut_x_list)
            result_dict['peak_global_pos'].append(peak_global_pos)

            continue
        return result_dict

    def plot_results(self, raw_sig, result_dict, window_expand_size = 40):
        '''Visualize result diction.'''
        p_wave_length = self.p_wave_length
        fs = self.fs
        raw_sig = self.raw_sig
        r_detector = DPI_QRS_Detector()

        for ind in xrange(0, len(result_dict['gpos'])):
            cut_x_list = result_dict['segment_range'][ind]
            gamp = result_dict['gamp'][ind]
            gsigma = result_dict['gsigma'][ind]
            gpos = result_dict['gpos'][ind]
            hermit_coefs = result_dict['hermit_coefs'][ind]
            peak_global_pos = result_dict['peak_global_pos'][ind]

            len_sig = cut_x_list[1] - cut_x_list[0]
            sig_seg = raw_sig[cut_x_list[0]:cut_x_list[1]]
            sig_seg = r_detector.HPF(sig_seg, fs = fs, fc = 3.0)

            fitting_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    gamp,
                    hermit_coefs)
            baseline_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    0,
                    hermit_coefs)
            plt.figure(1)
            plt.clf()
            plt.plot(sig_seg, label = 'ECG')
            plt.plot(fitting_curve, label = 'fitting curve')
            plt.plot(baseline_curve,
                    linestyle = '--',
                    alpha = 0.3, lw = 2,
                    label = 'baseline curve')
            plt.plot(gpos, fitting_curve[gpos], 'r^', markersize = 12)
            # plt.plot(gpos + gsigma, fitting_curve[gpos + gsigma],
                    # 'r^', markersize = 12)
            plt.legend()
            plt.grid(True)
            plt.show(block = False)

            # Plot fitting curve
            plt.figure(1)
            plt.plot(raw_sig, label = 'ECG')
            plt.plot(peak_global_pos, raw_sig[peak_global_pos],
                    'ro', markersize = 12,label = 'P peak')
            plt.plot(xrange(cut_x_list[0], cut_x_list[1]),
                    fitting_curve,
                    linewidth = 2, color = 'orange', alpha = 1,
                    label = 'Fitting curve')
            plt.plot(xrange(cut_x_list[0], cut_x_list[1]),
                    baseline_curve,
                    linewidth = 3, color = 'black', alpha = 0.3,
                    label = 'Baseline')
            plt.xlim((cut_x_list[0] - window_expand_size,
                cut_x_list[1] + window_expand_size))
            plt.title('ECG %s (Peak %d)' % ('signal', ind))
            plt.legend()
            plt.grid(True)
            plt.show()

def TEST():
    '''Test code for PDelineator.'''
    qt = QTloader()
    sig = qt.load('sel31')
    raw_sig = sig['sig'][1000:2000]
    
    pd = PDelineator(raw_sig, fs = 250.0)
    result = pd.run()
    pd.plot_results(raw_sig, result)

def TEST2():
    '''Test code for PDelineator.'''
    with open('./diagnosis_info.json', 'r') as fin:
        dinfo = json.load(fin)
        
    tested_file_list = list()
    testing_count = 0
    record_index = 0
    target_index = 2
    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            testing_count += 1
            print 'Progress:', testing_count
            print 'Diagnosis:', diagnosis_text
            if target_index > testing_count:
                print 'Target index is %d, skipping...' % target_index
                continue

            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name
            mat_file_name += '*.mat'
            result_json_file_path = os.path.join(
                    current_folder, record_ID + '_results.json')
            
            # Load mat file
            target_mat_file_list = glob.glob(
                    os.path.join(current_folder, mat_file_name))
            data = sio.loadmat(target_mat_file_list[0])
            sig = np.squeeze(data['aVR'])
            raw_sig = sig
            pd = PDelineator(raw_sig, fs = 500.0)
            debug_info = dict()
            debug_info['plot_title'] = u'aVR:%s' % diagnosis_text
            debug_info['plot_xlabel'] = record_ID
            result = pd.run(debug_info = debug_info)
            # pd.plot_results(raw_sig, result)

if __name__ == '__main__':
    # TEST()
    TEST2()

# ====Write to json====
# For display_results.py
# with open('./tmp.pkl', 'w') as fout:
    # pickle.dump(result_dict, fout)
