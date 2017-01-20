#encoding:utf8
import os
import sys
import time
import codecs

import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import numpy as np
import pdb
import math
import glob
import json


zn_font = matplotlib.font_manager.FontProperties(fname = '/usr/share/fonts/truetype/simsun.ttc')

class DiagnosisClass(object):
    def __init__(self, result_file_path, record_ID, fs = 500.0):
        '''Diagnosis Common Diseases.'''
        self.fs = fs
        with open(result_file_path, 'r') as fin:
            self.results = json.load(fin)
            self.results.sort(key = lambda x: x[0])
        # Load signal mat file
        mat_file_list = glob.glob(os.path.join(
            os.path.split(result_file_path)[0], record_ID + '*.mat'))
        if len(mat_file_list) == 0:
            raise Exception('Signal mat file does not exist!')
        self.sig_data = sio.loadmat(mat_file_list[0])
        for key in self.sig_data.keys():
            self.sig_data[key] = np.squeeze(self.sig_data[key])
        

        self.labels = set([x[1] for x in self.results])
        self.result_dict = dict()
        for label in self.labels:
            self.result_dict[label] = [x[0] for x in filter(lambda x:x[1] == label, self.results)]

        self.diagnosis_text = ""


    def plot(self, lead_name = 'II'):
        '''Plot ECG waveform with detected wave boundaries.'''
        plt.figure(1)
        plt.clf()
        plt.plot(self.sig_data[lead_name], label = 'ECG %s' % lead_name)
        for label in self.labels:
            pos_list = self.result_dict[label]
            amp_list = [self.sig_data[lead_name][x] for x in pos_list]
            plt.plot(pos_list, amp_list, 'o', label = label, markersize = 12,
                    alpha = 0.5)
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_diagnosis(self, abnormal_only = False):
        '''Diagnosis all the diseases.'''
        diagnose_functions = [self.Diag_sinus_bradycadia,
                self.Diag_fangchan,
                self.Diag_shisu,
                self.Diag_fangpu_fangsu,
                self.Diag_premature_beat,
                self.Diag_sinus_tachycardia,
                self.Diag_Arrithmia]
        if abnormal_only == False:
            diagnose_functions.append(self.Diag_Normal)
        # or
        for func in diagnose_functions:
            if func():
                return True


    def Diag_sinus_bradycadia(self):
        '''Daignosis(too slow).'''
        if 'R' not in self.result_dict:
            return True
        if len(self.result_dict['R']) < 2:
            return True
            # raise Exception('Not enough R wave detected!')

        r_poslist = self.result_dict['R']

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        if avg_RR < 1e-6:
            raise Exception('Average R-R interval is zero!')
            
        avg_bpm = float(self.fs * 60.0) / avg_RR

        if avg_bpm <= 60:
            return True
        else:
            return False

    def Diag_fangchan(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]

        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        high_heart_rate_count = len([x for x in filter(lambda x: x > 100, bpm_list)])
        # Must be sinus_tachycardia
        if avg_bpm < 100 and high_heart_rate_count < 3:
            # print bpm_list
            # print np.average(bpm_list[1:len(bpm_list) - 1])
            # print len([x for x in filter(lambda x: x > 100, bpm_list)])
            # print len(bpm_list)
            # print 'Not fast heart rate:', avg_bpm
            return False
        # Narrow
        qrs_width_list = list()
        pre_q_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        for ind in xrange(0, len_results):
            pos, label, x = self.results[ind]
            if label == 'Ronset':
                pre_q_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                if pre_r_pos != -1:
                    pre_r_pos = -1
                    pre_q_pos = -1
                else:
                    pre_r_pos = pos
            elif label == 'Roffset':
                if pre_q_pos != -1 and pre_r_pos != -1:
                    qrs_width_list.append(pos - pre_q_pos)
        
        qrs_width_list = [x / self.fs * 1000.0 for x in qrs_width_list]
        
        # (Both wide and narrow) Regular rhythm
        if np.std(bpm_list) > 10 and max(bpm_list) - min(bpm_list) > 40:
            return True
        else:
            # print 'Std:',np.std(bpm_list)
            return False

    def narrowQRS(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]

        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        high_heart_rate_count = len([x for x in filter(lambda x: x > 100, bpm_list)])
        # Must be sinus_tachycardia
        if avg_bpm < 100 and high_heart_rate_count < 3:
            # print bpm_list
            # print np.average(bpm_list[1:len(bpm_list) - 1])
            # print len([x for x in filter(lambda x: x > 100, bpm_list)])
            # print len(bpm_list)
            # print 'Not fast heart rate:', avg_bpm
            return False
        # Narrow
        qrs_width_list = list()
        pre_q_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        for ind in xrange(0, len_results):
            pos, label, x = self.results[ind]
            if label == 'Ronset':
                pre_q_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                if pre_r_pos != -1:
                    pre_r_pos = -1
                    pre_q_pos = -1
                else:
                    pre_r_pos = pos
            elif label == 'Roffset':
                if pre_q_pos != -1 and pre_r_pos != -1:
                    qrs_width_list.append(pos - pre_q_pos)
        
        qrs_width_list = [x / self.fs * 1000.0 for x in qrs_width_list]
        
        if np.average(qrs_width_list) < 120:
            return True
        else:
            print 'QRS width average:', np.average(qrs_width_list)
            return False

    def wideQRS(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]

        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        high_heart_rate_count = len([x for x in filter(lambda x: x > 100, bpm_list)])
        # Must be sinus_tachycardia
        if avg_bpm < 100 and high_heart_rate_count < 3:
            # print bpm_list
            # print np.average(bpm_list[1:len(bpm_list) - 1])
            # print len([x for x in filter(lambda x: x > 100, bpm_list)])
            # print len(bpm_list)
            # print 'Not fast heart rate:', avg_bpm
            return False
        # Narrow
        qrs_width_list = list()
        pre_q_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        for ind in xrange(0, len_results):
            pos, label, x = self.results[ind]
            if label == 'Ronset':
                pre_q_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                if pre_r_pos != -1:
                    pre_r_pos = -1
                    pre_q_pos = -1
                else:
                    pre_r_pos = pos
            elif label == 'Roffset':
                if pre_q_pos != -1 and pre_r_pos != -1:
                    qrs_width_list.append(pos - pre_q_pos)
        
        qrs_width_list = [x / self.fs * 1000.0 for x in qrs_width_list]
        
        if np.average(qrs_width_list) >= 120:
            return True
        else:
            return False

    def Diag_shisu(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]

        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        high_heart_rate_count = len([x for x in filter(lambda x: x > 100, bpm_list)])
        # Must be sinus_tachycardia
        if avg_bpm < 100 and high_heart_rate_count < 3:
            # print bpm_list
            # print np.average(bpm_list[1:len(bpm_list) - 1])
            # print len([x for x in filter(lambda x: x > 100, bpm_list)])
            # print len(bpm_list)
            # print 'Not fast heart rate:', avg_bpm
            return False
        # Narrow
        qrs_width_list = list()
        pre_q_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        for ind in xrange(0, len_results):
            pos, label, x = self.results[ind]
            if label == 'Ronset':
                pre_q_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                if pre_r_pos != -1:
                    pre_r_pos = -1
                    pre_q_pos = -1
                else:
                    pre_r_pos = pos
            elif label == 'Roffset':
                if pre_q_pos != -1 and pre_r_pos != -1:
                    qrs_width_list.append(pos - pre_q_pos)
        
        qrs_width_list = [x / self.fs * 1000.0 for x in qrs_width_list]
        
        # Should be wide QRS
        if np.average(qrs_width_list) < 120:
            return False
        else:
            pass
            # print 'Is wide QRS!'

        # Regular rhythm
        if np.std(bpm_list) > 30:
            # print 'Std:', np.std(bpm_list)
            return False
        else:
            pass
            # print 'Is Regular!'

        # P-R ratio
        if len(self.result_dict['P']) < 5:
            return False

        PR_ratio = float(len(self.result_dict['P'])) / len(self.result_dict['R'])

        # P bpm 
        p_poslist = self.result_dict['P']
        if len(p_poslist) < 2:
            raise Exception('Not enough P wave detected!')

        PP_period_list = [x[1] - x[0] for x in zip(p_poslist, p_poslist[1:])]
        if min(PP_period_list) < 1e-6:
            raise Exception('Average P-P interval is zero!')

        p_bpm_list = [float(self.fs * 60.0) / x for x in PP_period_list]
        p_avg_bpm = float(sum(p_bpm_list)) / len(p_bpm_list)
        
        if self.PR_1b1() == False:
            return p_avg_bpm < avg_bpm
        else:
            return True

    def PR_1b1(self):
        '''Check P-R ratio 1 : 1.'''
        if len(self.result_dict['R']) == 0:
            return len(self.result_dict['P']) < 2

        PR_ratio = float(len(self.result_dict['P'])) / len(self.result_dict['R'])
        if abs(PR_ratio - 1.0) < 0.2:
            return True
        else:
            return False

    def Diag_fangpu_fangsu(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]

        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        high_heart_rate_count = len([x for x in filter(lambda x: x > 100, bpm_list)])
        # Must be sinus_tachycardia
        if avg_bpm < 100 and high_heart_rate_count < 3:
            # print bpm_list
            # print np.average(bpm_list[1:len(bpm_list) - 1])
            # print len([x for x in filter(lambda x: x > 100, bpm_list)])
            # print len(bpm_list)
            # print 'Not fast heart rate:', avg_bpm
            return False
        # Narrow
        qrs_width_list = list()
        pre_q_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        for ind in xrange(0, len_results):
            pos, label, x = self.results[ind]
            if label == 'Ronset':
                pre_q_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                if pre_r_pos != -1:
                    pre_r_pos = -1
                    pre_q_pos = -1
                else:
                    pre_r_pos = pos
            elif label == 'Roffset':
                if pre_q_pos != -1 and pre_r_pos != -1:
                    qrs_width_list.append(pos - pre_q_pos)
        
        qrs_width_list = [x / self.fs * 1000.0 for x in qrs_width_list]
        
        # Judge Wide QRS
        if np.average(qrs_width_list) >= 120:
            is_wide_QRS = True
        else:
            is_wide_QRS = False

        # Regular rhythm
        if np.std(bpm_list) > 30:
            # print 'Std:', np.std(bpm_list)
            return False
        else:
            pass
            # print 'Is Narrow Regular!'

        # P visible
        if len(self.result_dict['P']) < 5:
            return False

        # P bpm 
        p_poslist = self.result_dict['P']
        if len(p_poslist) < 2:
            raise Exception('Not enough P wave detected!')

        PP_period_list = [x[1] - x[0] for x in zip(p_poslist, p_poslist[1:])]
        if min(PP_period_list) < 1e-6:
            raise Exception('Average P-P interval is zero!')

        p_bpm_list = [float(self.fs * 60.0) / x for x in PP_period_list]
        p_avg_bpm = float(sum(p_bpm_list)) / len(p_bpm_list)
        
        if is_wide_QRS:
            # Is 1:1
            if self.PR_1b1():
                return False

        return p_avg_bpm > avg_bpm
        

    def Diag_premature_beat(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]
        if min(RR_period_list) < 1e-6:
            raise Exception('Average R-R interval is zero!')
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]

        average_bpm = np.average(bpm_list)
        if average_bpm >= 100 or average_bpm <= 60:
            return False

        len_bpm = len(bpm_list)
        continous_premature = 0
        max_continous_premature = 0
        for ind in xrange(0, len_bpm):
            current_bpm = bpm_list[ind]
            if current_bpm >= 1.15 * average_bpm:
                continous_premature += 1
                if continous_premature > max_continous_premature:
                    max_continous_premature = continous_premature
            else:
                continous_premature = 0

        # Too fast:sinus_tachycardia
        if max_continous_premature > 2:
            return False
        return max_continous_premature > 0
                

    def Diag_sinus_tachycardia(self):
        '''Daignosis(too fast).'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]
        bpm_list = [float(self.fs * 60.0) / x for x in RR_period_list]
        
        avg_RR = float(sum(RR_period_list)) / len(RR_period_list)

        if avg_RR < 1e-6:
            raise Exception('Average R-R interval is zero!')
            
        avg_bpm = math.ceil(float(self.fs * 60.0) / avg_RR)

        # print bpm_list
        # print 'average bpm:', avg_bpm

        if avg_bpm >= 100:
            return True
        else:
            return False
        
    def GetQrsSum(self, raw_sig, q_index, r_index, s_index, MaxSearchLengthMs = 10):
        '''Get sum of amplitude of qrs for degree calculation.'''
        MaxSearchIndexLen = MaxSearchLengthMs / 1000.0 * self.fs
        ans = 0
        # q
        left_bound = int(max(0, q_index - MaxSearchIndexLen))
        right_bound = int(min(r_index, q_index + MaxSearchIndexLen))
        ans += np.min(raw_sig[left_bound:right_bound + 1])
        
        # r
        left_bound = int(max(q_index, r_index - MaxSearchIndexLen))
        right_bound = int(min(s_index, r_index + MaxSearchIndexLen))
        ans += np.max(raw_sig[left_bound:right_bound + 1])
        
        # s
        left_bound = int(max(r_index, s_index - MaxSearchIndexLen))
        right_bound = int(min(len(raw_sig) - 1, s_index + MaxSearchIndexLen))
        ans += np.max(raw_sig[left_bound:right_bound + 1])
        
        return ans

    def GetQrsDegree(self, q_index, r_index, s_index, MaxSearchLengthMs = 10):
        '''Get Degree.'''


        # Get sum of QRS in II
        pos_list = (q_index, r_index, s_index)
        raw_sig = self.sig_data['II']
        sumII = self.GetQrsSum(raw_sig, q_index, r_index, s_index, MaxSearchLengthMs = MaxSearchLengthMs)

        # Get sum of QRS in I
        raw_sig = self.sig_data['I']
        sumI = self.GetQrsSum(raw_sig, q_index, r_index, s_index, MaxSearchLengthMs = MaxSearchLengthMs)
        sum_avF = (sumII + sumI) / 2.0

        if abs(sumI) < 1e-6:
            theta = math.pi / 2.0
        else:
            # value of tan_theta
            atan_x = 2.0 / math.sqrt(3.0) * (float(sumII) / sumI - 0.5)
        
            if (sumI > 0 and sum_avF > 0):
                theta = math.atan(abs(atan_x))
            elif (sumI > 0 and sum_avF < 0):
                theta = -math.atan(abs(atan_x))
            elif (sumI < 0 and sum_avF > 0):
                theta = math.pi - math.atan(abs(atan_x))
            else:
                theta = math.pi + math.atan(abs(atan_x))

            theta = theta / math.pi * 180.0

        # print 'theta = ', theta

        # amplist = [raw_sig[x] if x < len(raw_sig) else 0 for x in pos_list]
        # plt.plot(raw_sig)
        # plt.plot(pos_list, amplist, 'ro', markersize = 12)
        # plt.show()
        # pdb.set_trace()
        return theta

    def Diag_DegreeList(self, MaxQRSGapMs = 500, SearchLengthMs = 10):
        '''Get Diagnosis Degree.'''
        pre_ronset_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        
        degree_list = list()

        for ind in xrange(0, len_results):
            pos, label, xx = self.results[ind]
            if label == 'Ronset':
                pre_ronset_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                pre_r_pos = pos
            elif label == 'Roffset':
                if (pre_ronset_pos != -1 and
                        pre_r_pos != -1 and
                        pos - pre_ronset_pos < MaxQRSGapMs / 1000.0 * self.fs):
                    # Is a Ronset-Roffset pair
                    # Search for actual positions of (q, r, s)
                    degree_list.append(self.GetQrsDegree(pre_ronset_pos, pre_r_pos, pos))
        return degree_list
                    
                    
    def Diag_Normal(self, debug_info = dict(),
            height_threshold = 900.0,
            rr_similar_threshold = 50.0,
            rr_max_fitting_difference_threshold = 200.0):
        '''Diagnosis.'''
        localpeak_r_pos_list = list()
        def cri_Qrs( width_threshold = 0.12,
                extreamly_bad_threshold = 500.0):
            '''Height of R wave must larger than 500(ZN data).
            Returns:
                True: Qrs height is normal.
                False: Qrs height is too low.
            '''
            result_dict = self.result_dict
            raw_sig = self.sig_data['II']
            R_list = result_dict['R']
            qrs_width = width_threshold * self.fs

            r_heights = list()
            for ind in xrange(0, len(R_list)):
                left = int(max(0, R_list[ind] - qrs_width/ 2.0))
                right = int(min(len(raw_sig), R_list[ind] + qrs_width/ 2.0))
                # Get r_height
                r_height = np.max(raw_sig[left:right])
                r_heights.append(r_height)
                localpeak_r_pos_list.append(np.argmax(raw_sig[left:right]) + left)

            if 'debug_plot' in debug_info:
                print '*' * 10
                print 'mean r_heights:', np.mean(r_heights)
                print 'r_height threshold:', height_threshold
                print '*' * 10

                # Plot R peak list
                # plt.figure(1)
                # plt.clf()
                # plt.plot(raw_sig)
                # amp_list = [raw_sig[x] for x in localpeak_r_pos_list]
                # plt.plot(localpeak_r_pos_list, amp_list, 'ro', markersize = 12)
                # plt.show(block = False)
                # plt.grid(True)
                # pdb.set_trace()

            if np.mean(r_heights) < height_threshold:
                return False
            else:
                return True

        def cri_RRsimilarity(score_threshold = 8.0, subsample_step = 5,
                iter_count = 1000,
                burn_count =500):
            '''Evaluate similarity of the waveform between R-R interval.'''
            def remove_edge_QRS(sig_in):
                sig_out = list(sig_in[:])
                while len(sig_out) > 0:
                    max_val = max(sig_out)
                    if sig_out[0] == max_val:
                        del sig_out[0]
                    elif len(sig_out) > 0 and sig_out[-1] == max_val:
                        del sig_out[-1]
                    else:
                        break
                return sig_out

            from rr_mcmc.wave_comparator import WaveComparator
            if 'R' not in self.result_dict or len(self.result_dict['R']) < 2:
                print 'Warning: record have no R detected, skipping similarity check...'
                return True

            raw_sig = self.sig_data['II']
            r_poslist = localpeak_r_pos_list
            len_r_poslist = len(r_poslist)

            score_list = list()
            max_fitting_difference_list = list()
            # Compare the similarity between nearby R-R region.
            for ind in xrange(0, len_r_poslist - 2):
                sig1 = raw_sig[r_poslist[ind]:r_poslist[ind + 1]][0::subsample_step]
                sig2 = raw_sig[r_poslist[ind + 1]:r_poslist[ind + 2]][0::subsample_step]

                sig1 = remove_edge_QRS(sig1)
                sig2 = remove_edge_QRS(sig2)

                sig1 = np.array(sig1, dtype = np.float32)
                sig2 = np.array(sig2, dtype = np.float32)

                wcomp = WaveComparator(sig1[:], sig2[:], 500.0 / subsample_step)

                if 'debug_plot' in debug_info:
                    max_difference = wcomp.compare_gaussian(
                        iter_count = iter_count,
                        burn_count = burn_count)
                    score = abs(wcomp.gamp)
                    score_list.append(score)
                    max_fitting_difference_list.append(max_difference)
                    print '*' * 10
                    print '[debug score]=', score
                    print '*' * 10

                    wcomp.plot()

                    # import rr_mcmc.rr_wave_model as rr_wave_model
                    # plt.clf()
                    # plt.plot(sig1 / 120.0, label = 'sig1')
                    # plt.plot(sig2 / 120.0, label = 'sig2')
                    # # Plot fitting curve
                    # comp_length = min(len(sig1), len(sig2))
                    # p_wave_list = np.zeros(int(comp_length),)
                    # for level, coef in zip(xrange(0,6), wcomp.hc_list):
                        # p_wave_list += rr_wave_model.HermitFunction(level,
                                # int(comp_length)) * coef
                    # plt.plot(p_wave_list, label = 'Difference complementary curve')
                    # plt.legend()
                    # plt.grid(True)
                    # if 'plot_title' in debug_info:
                        # plt.title(debug_info['plot_title'],
                                # fontproperties = zn_font, size = 'x-large')
                    # else:
                        # plt.title('compare sig1 and sig2')
                    # plt.show(block = False)
                    # pdb.set_trace()
                
                else:
                    max_difference = wcomp.compare_gaussian(
                        iter_count = iter_count,
                        burn_count = burn_count)
                    score = abs(wcomp.gamp)
                    score_list.append(score)
                    max_fitting_difference_list.append(max_difference)

                    print '[score]', score

            if np.max(score_list[3:len(score_list) - 1]) > rr_similar_threshold:
                return False
            elif (np.max(max_fitting_difference_list[3:len(score_list) - 1]) >
                    rr_max_fitting_difference_threshold):
                return False

            return True
            
        if self.run_diagnosis(abnormal_only = True):
            return False
        else:
            cri_functions = [cri_Qrs, cri_RRsimilarity]
            for cri_func in cri_functions:
                if cri_func() == False:
                    return False
            return True

    def Diag_Arrithmia(self):
        '''Diagnosis.'''
        if 'R' not in self.result_dict:
            return True
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]
        
        diag_diff = max(RR_period_list) - min(RR_period_list)

        if diag_diff / self.fs * 1000.0 >= 120:
            return True
        else:
            return False
        



def ShowSignal(folder_path, record_ID):
    '''Plot signal.'''
    mat_file = glob.glob(os.path.join(folder_path, record_ID + '*.mat'))
    if len(mat_file) == 0:
        print 'No mat file matching current ID:', record_ID
        return
    import scipy.io as sio
    sig = sio.loadmat(mat_file[0])
    raw_sig = np.squeeze(sig['II'])
    plt.plot(raw_sig)
    plt.show()
    
    
    
def Test():
    '''Test funcion for DiagnosisClass.'''
    with codecs.open('./diagnosis_info.json', 'r', 'utf8') as fin:
        dinfo = json.load(fin)

    # debug
    debug_count = 5
    TP_count = 0
    FN_count = 0
    FP_count = 0
    total_count = 0

    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name

            # Load results
            result_file_path = os.path.join(current_folder, record_ID + '_results.json')
            # Result file not exist
            if os.path.exists(result_file_path) == False:
                continue

            diag = DiagnosisClass(result_file_path, record_ID)
            if u'电轴' not in diagnosis_text or record_ID.startswith('TJ'):
                continue

            print 'diagnosis:', diagnosis_text
            print file_path
            degrees = diag.Diag_DegreeList()
            print 'mean degree:', np.nanmean(degrees)
            pdb.set_trace()

            # Statistics
            total_count += 1

            # Premature Beat
            if u'早搏' in diagnosis_text:
                if diag.Diag_premature_beat():
                    TP_count += 1
                else:
                    FN_count += 1
            elif diag.Diag_premature_beat():
                FP_count += 1

            # if u'不齐' in diagnosis_text:
                # if diag.Diag_Arrithmia():
                    # TP_count += 1
                # else:
                    # FN_count += 1
            # elif diag.Diag_Arrithmia():
                # FP_count += 1
                
                
            # if u'过速' in diagnosis_text:
                # if diag.Diag_sinus_tachycardia():
                    # # print '[过速检测到了！]'
                    # TP_count += 1
                # else:
                    # FN_count += 1
                    # print '@' * 10
                    # print diagnosis_text
            # elif diag.Diag_sinus_tachycardia():
                # FP_count += 1


            # if u'过缓' in diagnosis_text:
                # if diag.Diag_sinus_bradycadia():
                    # print '[过缓检测到了！]'
                    # TP_count += 1
                # else:
                    # FN_count += 1
                    # print '@' * 10
                    # print diagnosis_text
            # elif diag.Diag_sinus_bradycadia():
                # FP_count += 1

            # if diag.wideQRS():
                # print '\n'
                # print '*' * 10
                # print 'Wide QRS!'
                # print '*' * 10
                # TP_count += 1

            # if diag.narrowQRS():
                # FN_count += 1

            # @@@@@@@@@心房颤动@@@@@@@@@@
            # if u'心房颤动' in diagnosis_text:
                # if diag.Diag_fangchan():
                    # # print 'Diagnosis:', diagnosis_text
                    # # print '[心房颤动检测到了！]'
                    # TP_count += 1
                # else:
                    # FN_count += 1
                    # print '@' * 10
                    # print diagnosis_text
                    # print 'file_path:', file_path
                # # ShowSignal(current_folder, record_ID)
            # elif diag.Diag_fangchan():
                # FP_count += 1
                
            # @@@@@@@@@室性心动过速@@@@@@@@@@
            # if u'室性心动过速' in diagnosis_text:
                # if diag.Diag_shisu():
                    # print 'Diagnosis:', diagnosis_text
                    # print '[室性心动过速检测到了！]'
                    # TP_count += 1
                # else:
                    # FN_count += 1
                    # print '@' * 10
                    # print diagnosis_text
                    # print 'file_path:', file_path
            # elif diag.Diag_shisu():
                # FP_count += 1

            # @@@@@@@@@房性心动过速@@@@@@@@@@
            # if (u'房性心动过速' in diagnosis_text):
            # if (u'房性心动过速' in diagnosis_text or
                    # u'心房扑动' in diagnosis_text):
            # if (u'房性心动过速' in diagnosis_text or
                    # u'心房扑动' in diagnosis_text or
                    # u'心房颤动' in diagnosis_text):
                # if diag.Diag_fangpu_fangsu() or diag.Diag_fangchan():
                    # print 'Diagnosis:', diagnosis_text
                    # print '[房性心动过速检测到了！]'
                    # TP_count += 1
                # else:
                    # FN_count += 1
                    # print '@' * 10
                    # print diagnosis_text
                    # print 'file_path:', file_path
            # elif diag.Diag_fangpu_fangsu() or diag.Diag_fangchan():
                # FP_count += 1
                
            # print '@' * 10
            # print diagnosis_text
            # if diag.Diag_Arrithmia():
                # print 'XXXX[Is Arrithmia!]XXXX'
            
            debug_count -= 1
            if debug_count <= 0:
                break
    print
    print '#' * 20
    print 'TP:', TP_count
    print 'FN:', FN_count
    print 'FP:', FP_count
    print 'Total:', total_count

def Test_Normal(rr_similar_threshold, debug_count = 1000,
        debug_info_Test = dict()):
    '''Test funcion for Normal DiagnosisClass.'''
    with codecs.open('./diagnosis_info.json', 'r', 'utf8') as fin:
        dinfo = json.load(fin)

    # debug
    if 'only_test_json' in debug_info_Test:
        with open(debug_info_Test['only_test_json'], 'r') as fin:
            FN_list, FP_list, diagnosis_statistics = json.load(fin)
        target_ID_set = set()
        for folder_path, record_ID, diagnosis_text in FP_list:
            target_ID_set.add(folder_path + record_ID)
    
    TP_count = 0
    FN_count = 0
    FP_count = 0
    total_count = 0

    # Record file names for debug.
    FN_file_list = list()
    FP_file_list = list()

    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name

            # Load results
            result_file_path = os.path.join(current_folder, record_ID + '_results.json')
            # Result file not exist
            if os.path.exists(result_file_path) == False:
                continue


            # Skipping TiJian records
            if record_ID.startswith('TJ'):
                continue
            # debug
            if 'only_test_json' in debug_info_Test:
                if record_ID != u'MEDEXS120161026094556734':
                    continue
                # if current_folder + record_ID not in target_ID_set:
                    # continue

            diag = DiagnosisClass(result_file_path, record_ID)
            # Debug Plot
            print 'debugging...'
            debug_info = dict()
            debug_info['debug_plot'] = True
            debug_info['plot_title'] = diagnosis_text
            diag.Diag_Normal(rr_similar_threshold = rr_similar_threshold, debug_info = debug_info)

            print 'diagnosis:', diagnosis_text
            print file_path

            # Statistics
            total_count += 1

            # Premature Beat
            if u'正常' in diagnosis_text:
                if diag.Diag_Normal(rr_similar_threshold = rr_similar_threshold):
                    TP_count += 1
                else:
                    FN_count += 1
                    FN_file_list.append((current_folder, record_ID))
                    # Debug 
                    # debug_info = dict()
                    # debug_info['debug_plot'] = True
                    # diag.Diag_Normal(rr_similar_threshold = rr_similar_threshold, debug_info = debug_info)

            elif diag.Diag_Normal(rr_similar_threshold = rr_similar_threshold):
                diag_phrase_list = diagnosis_text.split('\n')
                can_skip = True
                for d_phrase in diag_phrase_list:
                    if d_phrase not in [u'窦性心律', u'顺钟向转位',
                            u'逆钟向转位']:
                        d_phrase = d_phrase.strip()
                        if (d_phrase.startswith(u'电轴左偏') or 
                                d_phrase.startswith(u'电轴右偏')):
                            continue
                        can_skip = False
                        break
                if can_skip == False:
                    FP_count += 1
                    degrees = diag.Diag_DegreeList()
                    print 'mean degree:', np.nanmean(degrees)

                    FP_file_list.append((current_folder, record_ID))
                    # Stop and plot signal
                    # Debug 
                    if ('skip_head' in debug_info_Test and
                            total_count <= debug_info_Test['skip_head']):
                        continue
                    debug_info = dict()
                    debug_info['debug_plot'] = True
                    debug_info['plot_title'] = diagnosis_text
                    diag.Diag_Normal(rr_similar_threshold = rr_similar_threshold, debug_info = debug_info)
                else:
                    TP_count += 1
                

            if total_count > debug_count:
                break
            if total_count % 50 == 0:
                print
                print '#' * 20
                print 'TP:', TP_count
                print 'FN:', FN_count
                print 'FP:', FP_count
                print 'TN:', total_count - TP_count - FN_count - FP_count
                print 'Total:', total_count

                if 'write2json' in debug_info_Test:
                    # Write to json file
                    diagnosis_statistics = [FN_file_list, FP_file_list,
                            dict(TP=TP_count,
                                FN=FN_count,
                                FP=FP_count,
                                TN=total_count - TP_count - FN_count - FP_count,
                                total=total_count)]
                    with codecs.open('./Diagnosis_Normal_Fails1.json', 'w', 'utf8') as fout:
                        json.dump(diagnosis_statistics, fout, indent = 4,
                                ensure_ascii = False)
                        print 'Misclassified records are written to Diagnosis_Normal_Fails.json'
            
    print
    print '#' * 20
    print 'TP:', TP_count
    print 'FN:', FN_count
    print 'FP:', FP_count
    print 'TN:', total_count - TP_count - FN_count - FP_count
    print 'Total:', total_count

    if 'write2json' in debug_info_Test:
        # Write to json file
        diagnosis_statistics = [FN_file_list, FP_file_list,
                dict(TP=TP_count,
                    FN=FN_count,
                    FP=FP_count,
                    TN=total_count - TP_count - FN_count - FP_count,
                    total=total_count)]
        with codecs.open('./Diagnosis_Normal_Fails1.json', 'w', 'utf8') as fout:
            json.dump(diagnosis_statistics, fout, indent = 4,
                    ensure_ascii = False)
            print 'Misclassified records are written to Diagnosis_Normal_Fails.json'


# Test()
start_time = time.time()
debug_info = dict()
# debug_info['write2json'] = True
debug_info['only_test_json'] = 'Diagnosis_Normal_Fails.json'
debug_info['skip_head'] = 0

Test_Normal(100.0, debug_count = 1000, debug_info_Test = debug_info)
print 'Testing time: %d seconds.' % (time.time() - start_time)
