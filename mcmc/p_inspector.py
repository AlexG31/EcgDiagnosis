#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pdb
from QTdata.loadQTdata import QTloader


def inspect(qt, rec_ind):
    '''Plot signal of 2 leads.'''
    reclist = qt.getreclist()
    sig = qt.load(reclist[rec_ind])
    sigd1 = sig['sig']
    sigd2 = sig['sig2']
    
    plt.plot(sigd1, label = 'ECG lead1')
    plt.plot(sigd2, label = 'ECG lead2')
    plt.legend()
    plt.title('ECG[%d] %s' % (rec_ind, reclist[rec_ind]))
    plt.xlim((0,1000))
    plt.grid(True)
    plt.show()


qt = QTloader()
for rec_ind in xrange(18, 105):
    inspect(qt, rec_ind)

    
