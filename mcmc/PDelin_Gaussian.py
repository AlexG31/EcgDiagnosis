#encoding:utf8
import os
import sys
import glob
import matplotlib.pyplot as plt
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


class PDelineator(object):
    def __init__(self, raw_sig, fs = 250.0, p_wave_lengthMs = 38 * 4):
        '''Delineator of P wave.'''
        # Time statistics
        # write to json
        # qt = QTloader()
        # reclist = qt.getreclist()
        # rec_ind = 21
        # sig = qt.load(reclist[rec_ind])
        self.raw_sig = raw_sig
        self.fs = fs
        self.p_wave_length = p_wave_lengthMs * fs / 1000.0

        # Plot 2 lead signal
        # plt.plot(raw_sig, label = 'lead2')
        # plt.plot(sig['sig'], label = 'lead1')
        # plt.title('%s' % reclist[rec_ind])
        # plt.show()


        r_detector = DPI_QRS_Detector()
        start_time = time.time()
        self.r_list = r_detector.QRS_Detection(raw_sig)
        print 'R detection time cost : %d seconds.' % (time.time() - start_time)

        # index of R wave
        result_dict = dict()
        # init
        result_dict['bs0'] = list()
        result_dict['bs1'] = list()
        result_dict['bs2'] = list()
        result_dict['bs3'] = list()
        result_dict['bs4'] = list()
        result_dict['P_amp'] = list()
        result_dict['P_pos'] = list()
        result_dict['P_dc'] = list()
        result_dict['P_sigma'] = list()
        result_dict['min_val'] = list()
        result_dict['max_val'] = list()
        result_dict['peak_global_pos'] = list()
        result_dict['segment_range'] = list()

        self.result_dict = result_dict

    def run(self, debug_info = dict()):
        '''Run delineation process for each R-R interval.'''

        r_list = self.r_list
        result_dict = self.result_dict
        fs = self.fs
        raw_sig = self.raw_sig
        p_wave_length = self.p_wave_length

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
            if len(sig_seg) <= 100 * fs / 250.0:
                print 'R-R interval is too short!'
                continue

            # plt.figure(2)
            # plt.plot(sig_seg, label = 'sig segment')
            # plt.title('Signal Segment')


            p_model = P_model_Gaussian.MakeModel(sig_seg, p_wave_length, fs = fs)
            M = MCMC(p_model)

            M.sample(iter = 2000, burn = 1000, thin = 10)


            amp = np.mean(M.trace('P_amp')[:])
            pos = np.mean(M.trace('P_pos')[:])
            P_dc = np.mean(M.trace('P_dc')[:])
            P_sigma = np.mean(M.trace('P_sigma')[:])



            # Fitting curve
            len_sig = p_model['len_sig']

            p_wave_list = P_model_Gaussian.GetGaussianPwave(p_wave_length,
                    amp, P_sigma, P_dc)
            fitting_sig = np.array(p_wave_list)


            bs0 = np.mean(M.trace('bs0')[:])
            bs1 = np.mean(M.trace('bs1')[:])
            bs2 = np.mean(M.trace('bs2')[:])
            bs3 = np.mean(M.trace('bs3')[:])
            bs4 = np.mean(M.trace('bs4')[:])
            baseline_kx = [bs0, bs1,bs2,bs3, bs4,]

            baseline_curve = np.zeros(len_sig,)
            # add baseline
            baseline_mat = P_model_Gaussian.GetBaselineMatrix(len_sig, fs)
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
            result_dict['P_amp'].append(amp)
            result_dict['P_pos'].append(pos)
            result_dict['P_dc'].append(P_dc)
            result_dict['P_sigma'].append(P_sigma)
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
        return result_dict

    def plot_results(self, raw_sig, result_dict):
        '''Visualize result diction.'''
        p_wave_length = self.p_wave_length
        fs = self.fs
        for ind in xrange(0, len(result_dict['bs0'])):
            cut_x_list = result_dict['segment_range'][ind]
            max_val = float(result_dict['max_val'][ind])
            min_val = float(result_dict['min_val'][ind])
            peak_global_pos = result_dict['peak_global_pos'][ind]
            bs0 = result_dict['bs0'][ind]
            bs1 = result_dict['bs1'][ind]
            bs2 = result_dict['bs2'][ind]
            bs3 = result_dict['bs3'][ind]
            bs4 = result_dict['bs4'][ind]
            P_amp = result_dict['P_amp'][ind]
            P_pos = result_dict['P_pos'][ind]
            P_sigma = result_dict['P_sigma'][ind]
            P_dc = result_dict['P_dc'][ind]

            peak_pos = peak_global_pos - cut_x_list[0]
            baseline_kx = [bs0, bs1,bs2,bs3, bs4,]

            whole_sig_left = max(cut_x_list[0] - 50, 0)
            whole_sig_bias = cut_x_list[0] - whole_sig_left
            whole_sig_right = min(cut_x_list[1] + 50, len(raw_sig))
            whole_sig = raw_sig[whole_sig_left:whole_sig_right]
            # Lead2
            whole_sig2 = np.array(raw_sig[whole_sig_left:whole_sig_right],
                    dtype = np.float32)
            if max_val - min_val > 1e-6:
                whole_sig = np.array(whole_sig, dtype = np.float32)
                whole_sig -= min_val
                whole_sig /= (max_val - min_val)
                # Lead2
                whole_sig2 = np.array(whole_sig2)
                whole_sig2 -= min_val
                whole_sig2 /= (max_val - min_val)

            # Fitting curve
            len_sig = cut_x_list[1] - cut_x_list[0]
            # P wave shape
            p_wave_list = P_model_Gaussian.GetGaussianPwave(p_wave_length,
                    P_amp, P_sigma, P_dc)
            fitting_sig = np.array(p_wave_list)
            # Baseline shape
            baseline_curve = np.zeros(len_sig,)
            baseline_mat = P_model_Gaussian.GetBaselineMatrix(len_sig, fs)
            part_p_wave = list()
            part_p_indexes = list()
            for baseline_list, kx in zip(baseline_mat, baseline_kx):
                baseline_curve += baseline_list * kx
            fitting_sig = np.copy(baseline_curve)
            for p_ind, p_val in enumerate(p_wave_list):
                sig_ind = p_ind - p_wave_length / 2.0 + P_pos
                sig_ind = int(sig_ind)
                # Out of scope
                if sig_ind >= len_sig or sig_ind < 0:
                    continue
                fitting_sig[sig_ind] += p_val
                part_p_wave.append(fitting_sig[sig_ind])
                part_p_indexes.append(p_ind)
            # Plot fitting curve
            plt.figure(1)
            plt.plot(whole_sig, label = 'ECG')
            # plt.plot(whole_sig2, label = 'Lead2')
            plt.plot(peak_pos + whole_sig_bias, whole_sig[peak_pos + whole_sig_bias],
                    'ro', markersize = 12,label = 'P pos')
            plt.plot(xrange(whole_sig_bias, whole_sig_bias + len(fitting_sig)),
                    fitting_sig,
                    linewidth = 2, color = 'orange', alpha = 1,
                    label = 'Fitting curve')
            plt.plot(xrange(whole_sig_bias, whole_sig_bias + len(baseline_curve)),
                    baseline_curve,
                    linewidth = 3, color = 'black', alpha = 0.3,
                    label = 'Baseline')
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
            result_json_file_path = os.path.join(current_folder, record_ID + '_results.json')
            
            # Load mat file
            target_mat_file_list = glob.glob(
                    os.path.join(current_folder, mat_file_name))
            data = sio.loadmat(target_mat_file_list[0])
            sig = np.squeeze(data['aVR'])
            raw_sig = sig
            pd = PDelineator(raw_sig, fs = 500.0)
            result = pd.run()
            pd.plot_results(raw_sig, result)

if __name__ == '__main__':
    # TEST()
    TEST2()

# ====Write to json====
# For display_results.py
# with open('./tmp.pkl', 'w') as fout:
    # pickle.dump(result_dict, fout)
