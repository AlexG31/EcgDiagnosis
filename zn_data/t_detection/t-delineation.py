#encoding:utf8
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pymc
from pymc import MCMC
import pdb
import importlib
from dpi.DPI_QRS_Detector import DPI_QRS_Detector

t_gaussian_model = importlib.import_module('t_detection.t-gaussian-model')

class TDelineator(object):
    def __init__(self, sig1, fs, mcmc_model = t_gaussian_model):
        '''Compare the similarity between ECG wave segments.'''
        self.sig1 = np.array(sig1, dtype = np.float32)
        self.fs = fs
        self.hc_list = list()
        self.mcmc_model = mcmc_model
        self.r_detector = DPI_QRS_Detector({'time_cost':True})

    def process_segment(self,
            filtered_sig,
            fs,
            qrs_pos_current,
            qrs_pos_next,
            step = 4,
            SJ_length = 20,
            ):
        '''Return processed T wave segment.'''
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

        segment_range = [qrs_pos_current[2], qrs_pos_next[0]]
        if (qrs_pos_current[2] + SJ_length < len(filtered_sig)):
            segment_range[0] = qrs_pos_current[2] + SJ_length

        raw_sig = filtered_sig[segment_range[0]: segment_range[1]][0::step]
        if len(raw_sig) == 0:
            return None
        remain_range = remove_edge_QRS(raw_sig)
        global_bias = segment_range[0] + remain_range[0]

        # Skip short signals
        if remain_range[1] - remain_range[0] <= 20:
            return None

        raw_sig = raw_sig[remain_range[0]:remain_range[1]]

        # HPF ECG segment
        raw_sig = self.r_detector.HPF(raw_sig, fs = fs, fc = 2.0)

        # Convert to np array
        raw_sig = np.array(raw_sig, dtype = np.float32)
        if np.max(raw_sig) - np.min(raw_sig) < 1e-6:
            print 'Warning: T wave segment is DC signal!'
            return None

        return (raw_sig, global_bias)

    def detect(self,
            filtered_sig,
            fs,
            qrs_pos_current,
            qrs_pos_next,
            iter_count = 500,
            burn_count = 250,
            ignore_edge_length = 5,
            step = 4):
        '''Detect T wave attributes.
            Note:
                Return None on short segments.
        '''
        raw_sig, global_bias = self.process_segment(filtered_sig, fs,
                qrs_pos_current, qrs_pos_next)
        if raw_sig is None:
            return None

        t_mcmc_model = t_gaussian_model.MakeModel(
                raw_sig,
                fs / step,
                ignore_edge_length = ignore_edge_length)

        mcmc = MCMC(t_mcmc_model)
        self.mcmc_object = mcmc

        mcmc.sample(iter = iter_count, burn = burn_count, thin = 10)

        # Compute Score
        common_length = len(raw_sig)
        gpos = np.mean(self.mcmc_object.trace('gaussian_start_position')[:])
        gsigma = np.mean(self.mcmc_object.trace('gaussian_sigma')[:])
        gamp = np.mean(self.mcmc_object.trace('gaussian_amplitude')[:])

        # complement step
        gpos *= step
        gsigma *= step

        results = dict()
        results['Tonset'] = int(gpos)
        results['Tpeak'] = int(gpos + gsigma / 2)
        results['Toffset'] = int(gpos + gsigma)
        results['global_bias'] = global_bias
        results['amplitude'] = gamp
        results['step'] = step

        return results


    def detection(self,
            ignore_edge_length = 5,
            iter_count = 500,
            burn_count = 250):
        '''Compare and Return the similarity score.'''

        sig1 = self.sig1
        rr_model = t_gaussian_model.MakeModel(self.sig1,
                self.fs,
                ignore_edge_length = ignore_edge_length)

        mcmc = MCMC(rr_model)
        self.mcmc_object = mcmc

        mcmc.sample(iter = iter_count, burn = burn_count, thin = 10)

        # Compute Score
        common_length = len(sig1)
        gpos = np.mean(self.mcmc_object.trace('gaussian_start_position')[:])
        gsigma = np.mean(self.mcmc_object.trace('gaussian_sigma')[:])
        gamp = np.mean(self.mcmc_object.trace('gaussian_amplitude')[:])

        # Ignore narrow Gaussian curves
        # if gsigma <= 7.9:
            # print 'Narrow Gaussian fitting curve ignored!'
            # gamp = 0.0


        hc_list = list()
        for ind in xrange(0, t_gaussian_model.HermitFunction_max_level):
            hc_mean = np.mean(self.mcmc_object.trace('hc%d' % ind)[:])
            hc_list.append(hc_mean)
        fitting_curve = t_gaussian_model.GetFittingCurve(common_length,
                gpos, gsigma,
                gamp,
                hc_list,
                )

        score = np.max(np.absolute(
            fitting_curve[ignore_edge_length:common_length - ignore_edge_length]))
        if ignore_edge_length * 2 >= common_length:
            score = 0.0
        else:
            score = np.sum(np.absolute(
                fitting_curve[ignore_edge_length:common_length - ignore_edge_length])) / (common_length - 2 * ignore_edge_length)
        
        print 'Maximum amplitude in the curve:', score
        print 'gaussian sigma = ', gsigma
        print 'gaussian amplitude = ', gamp
        self.gamp = gamp
        # print 'gaussian baseline = ', gbase
        # print 'score = ', max(abs(gamp), score)

        # score = max(abs(gamp), score)

        return score


    def GetTpos(self):
        '''Call this after calling detection() function.
            Returns:
                (T_onset, T_peak, T_offset) indexes.
        '''
        gpos = np.mean(self.mcmc_object.trace('gaussian_start_position')[:])
        gsigma = np.mean(self.mcmc_object.trace('gaussian_sigma')[:])

        gaussian_peak_pos = gpos + gsigma / 2.0
        T_onset = gpos
        T_offset = gpos + gsigma
        return (T_onset, gaussian_peak_pos, T_offset)
    def GetTamp(self):
        '''Call this after calling detection() function.
            Returns:
                gaussian amplitude of T wave.
        '''
        gamp = np.mean(self.mcmc_object.trace('gaussian_amplitude')[:])
        return gamp



    def trace_plot(self):
        '''MCMC Plot'''
        from pymc.Matplot import plot
        plot(self.mcmc_object)
        
    def plot(self):
        '''Plot mcmc result.'''
        sig1 = self.sig1
        common_length = len(sig1)
        gpos = np.mean(self.mcmc_object.trace('gaussian_start_position')[:])
        gsigma = np.mean(self.mcmc_object.trace('gaussian_sigma')[:])
        gamp = np.mean(self.mcmc_object.trace('gaussian_amplitude')[:])

        gaussian_peak_pos = gpos + gsigma / 2.0
        T_poslist = self.GetTpos()



        hc_list = list()
        for ind in xrange(0, t_gaussian_model.HermitFunction_max_level):
            hc_mean = np.mean(self.mcmc_object.trace('hc%d' % ind)[:])
            hc_list.append(hc_mean)
        fitting_curve = t_gaussian_model.GetFittingCurve(common_length,
                gpos, gsigma,
                gamp,
                hc_list,
                )
        fitting_curve_no_g = t_gaussian_model.GetFittingCurve(common_length,
                gpos, gsigma,
                0,
                hc_list,
                )

        score = np.max(np.absolute(fitting_curve))
        noise_sigma = np.mean(self.mcmc_object.trace('wave_shape_sigma')[:])
        print 'Maximum amplitude in the curve:', score
        print 'gaussian sigma = ', gsigma
        print 'gaussian amplitude = ', gamp
        # print 'gaussian baseline = ', gbase
        print '[noise sigma]:', noise_sigma
        print 'score(max(gamp, score)) = ', max(abs(gamp), score)

        plt.figure(1)
        plt.clf()

        sig1_seg = sig1[:]
        plt.plot(sig1_seg, label = 'Sig1')
        amp_list = [sig1_seg[x] for x in T_poslist]
        plt.plot(T_poslist, amp_list, 'mo', markersize = 12,
                label = 'T detection result')
        plt.plot(gaussian_peak_pos, fitting_curve[gaussian_peak_pos], 'r^', markersize = 12,
                label = 'Center of Gaussian wave')
        plt.plot(fitting_curve, label = 'Difference complementary curve')
        plt.plot(fitting_curve_no_g, label = 'Difference complementary curve')
        plt.legend()
        plt.grid(True)

        plt.show(block = False)
        pdb.set_trace()



def TestGaussian():
    '''Test Gaussian mcmc model'''
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

    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    reclist = qt.getreclist()
    recname = reclist[30]
    sig = qt.load(recname)
    # raw_sig = sig['sig2'][0:10000]
    raw_sig = sig['sig'][10000:20000]

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

    step = 2
    for ind in xrange(15, 25):
        # Qrs analysis
        r_pos = r_list[ind]
        qrs_pos, qrs_text = qrs_type.GetQrsType(filtered_sig,
                r_pos - 10, r_pos, r_pos + 10,
                debug_plot = True)

        print qrs_text
        print qrs_pos

        if qrs_pos[2] + 20 >= len(raw_sig):
            print 'Warning: qrs_pos exceeds boundary!'
            continue
        sig1 = raw_sig[qrs_pos[2] + 20:r_list[ind + 1]][0::step]
        remain_range = remove_edge_QRS(sig1)
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
        print 'T wave position = ', pos

        # show full signal
        bound_list = [r_list[ind],r_list[ind + 1]]
        plt.figure(2)
        plt.clf()
        plt.plot(raw_sig, label = 'ECG')
        plt.plot(filtered_sig, label = 'ECG after HPF')
        plt.plot(bound_list, [raw_sig[x] for x in bound_list], 'ro')
        plt.xlim((bound_list[0] - 40, bound_list[1] + 40))
        plt.legend()
        plt.title(recname)
        plt.grid(True)
        
        wcomp.plot()

if __name__ == '__main__':
    # Test()
    TestGaussian()
        
