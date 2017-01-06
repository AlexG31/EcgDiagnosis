#encoding:utf8

import os
import sys
import matplotlib.pyplot as plt
import json
from QTdata.loadQTdata import QTloader


def sample():
    '''ECG.'''
    qt = QTloader()

    sig = qt.load('sel100')
    raw_sig = sig['sig']
    plt.plot(raw_sig)
    plt.title('ECG Sample.')
    plt.show()

def CutSample():
    '''Cut Ecg segment to json.'''
    qt = QTloader()
    sig = qt.load('sel100')
    raw_sig = sig['sig']
    sig_seg = raw_sig[40:128]
    with open('./segment.json', 'w') as fout:
        json.dump(sig_seg, fout, indent = 4)
    

# sample()
CutSample()
