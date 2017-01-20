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

t_gaussian_model = importlib.import_module('T-gaussian-model')

class TDelineator(object):
    def __init__(self, sig1, fs, mcmc_model = t_gaussian_model):
        '''Compare the similarity between ECG wave segments.'''
        self.sig1 = np.array(sig1, dtype = np.float32)
        self.fs = fs
        self.hc_list = list()
        self.mcmc_model = mcmc_model

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
        gamp = np.mean(self.mcmc_object.trace('gaussian_amp')[:])

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


    def plot(self):
        '''Plot mcmc result.'''
        sig1 = self.sig1
        common_length = len(sig1)
        gpos = np.mean(self.mcmc_object.trace('gaussian_start_position')[:])
        gsigma = np.mean(self.mcmc_object.trace('gaussian_sigma')[:])
        gamp = np.mean(self.mcmc_object.trace('gaussian_amp')[:])



        hc_list = list()
        for ind in xrange(0, t_gaussian_model.HermitFunction_max_level):
            hc_mean = np.mean(self.mcmc_object.trace('hc%d' % ind)[:])
            hc_list.append(hc_mean)
        fitting_curve = t_gaussian_model.GetFittingCurve(common_length,
                gpos, gsigma,
                gamp,
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
        plt.plot(fitting_curve, label = 'Difference complementary curve')
        plt.legend()
        plt.grid(True)

        plt.show(block = False)
        pdb.set_trace()



def TestGaussian():
    '''Test Gaussian mcmc model'''
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

    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    sig = qt.load('sel34')
    # raw_sig = sig['sig2'][0:10000]
    raw_sig = sig['sig2'][10000:20000]

    from dpi.DPI_QRS_Detector import DPI_QRS_Detector

    print 'Start testing R waves with DPI.'
    debug_info = dict()
    debug_info['time_cost'] = True
    r_detector = DPI_QRS_Detector(debug_info = debug_info)
    r_list = r_detector.QRS_Detection(raw_sig, 250.0)

    for ind in xrange(15, 25):
        step = 2
        sig1 = raw_sig[r_list[ind]:r_list[ind + 1]][0::step]
        sig1 = remove_edge_QRS(sig1)

        # Convert to np array
        sig1 = np.array(sig1, dtype = np.float32)
        sig1 *= 1000.0
        sig1 -= np.max(sig1)

        wcomp = TDelineator(sig1, 250.0 / step, t_gaussian_model)
        pos = wcomp.detection(iter_count = 1000, burn_count = 500)
        print 'T wave position = ', pos

        # show full signal
        bound_list = [r_list[ind],r_list[ind + 1]]
        plt.figure(2)
        plt.clf()
        plt.plot(raw_sig)
        plt.plot(bound_list, [raw_sig[x] for x in bound_list], 'ro')
        plt.xlim((bound_list[0] - 40, bound_list[1] + 40))
        plt.grid(True)
        
        wcomp.plot()

if __name__ == '__main__':
    # Test()
    TestGaussian()
        
