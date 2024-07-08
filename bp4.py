#!/usr/bin/env python
#
# TODO:
# include version and or date in log string


import numpy as np
import matplotlib.pyplot as plt
plt.ion() #  interactive mode, so that show() doesn't block next plot
import copy
import random
from pathlib import Path
import textwrap as textwrap
import os
from multiprocessing import Pool
import sys, importlib
importlib.reload(sys.modules.get('runnet',sys))
from runnet import runnet
from dataclasses import dataclass

#from pudb import set_trace; set_trace()  # uncomment for pudb-debugger
#np.seterr(invalid='raise') # quit as soon as getting nan or similar

np.set_printoptions(precision=4)

def dict_to_string(dic,skiplist):
    # creates log-string from a dictionary or dataclass
    # skip vars in skiplist
    # for dataclasses, call on vars(dc), dict_to_string(vars(dc))
    outputstring = ""
    for key in dic:
        if key in skiplist:
            continue
        outputstring +="_%s=%s" % (key,dic[key])
    return outputstring

## wrapper function
@dataclass
class parameters:
    test: float =0
    def __init__(self):
        self.smalldatasetQ:  bool = False
        self.biasQ:          bool = True
        self.zeromeandataQ:  bool = True
        self.xtropyQ :       bool = False  # use crossentropy loss, otherwise use MSE
        self.learn_rate:     np.array = np.array([0.01, 0.01]) # per layer.
        self.gfun:           string = 'lru'

        self.compete_algol:  int = 7
        # if =0: no competition, =1: per neuron , =2: largest synaptic, =3: per synapse,
        # =4: fixed fraction of neurons, =5 fixed fraction of synapses
        # =6: output dependent fixed fraction. =7 completely fixed input layer
        # =10: alternating mask

        self.pcompete:       np.array = np.array([1.,1.]) #  \competition per layer. =1: no competition
        self.learnratescaleQ:bool = False

        self.Nhid:           int   = 200
        self.lam2reg:        float = 0.0 # L2 regularizer on weights

        self.si_w0:          np.array = np.array([0.03,0.03]) # initial Gaussian weights, better way see Klein.

        self.accuracy_goal:  float = 0.95 # exit when reached on test-set
        self.maxepochs :     float = 5# max epochs before giving up.
        self.testinterval:   int   = 2000
        self.nmasks:         int   = 0
        self.version:        int   = 2# version2, 3/2022. Return L0 and L1 energy, 
        self.settingstr:     string

########### START ###############
plt.close('all')

# settings to be omitted from settingstr:
skiplist=['smalldatasetQ','biasQ','ReLuQ','testinterval','settingstr','learnratescaleQ','lam2reg','accuracy_goal','nmasks']

# start MAIN

debugrunQ= False
if debugrunQ:
    param=parameters()
    param.pcompete[0]=1
    param.compete_algol = 7
    print(param)
    runnet(param)
    raise

# outerloop is used to have multiple curves in a plot
# we loop explicitly over this instead of using map.
#outerloopvarstr = "compete_algol"
#outerloopvals=[4,20,40]

outerloopvarstr = "si_w0"
outerloopvals=[0.01,0.02]
Npouter=len(outerloopvals)
print('outloopvals= ', outerloopvals)


if (Npouter > 1):
    print('bp4.py: outerloopvals = ', outerloopvals)
   # skiplist.append(outerloopvarstr)

fig,axs = plt.subplots(nrows = 3, ncols = 4, sharex='all')

for outerloopvar in outerloopvals:
    if (Npouter > 1):
        print(outerloopvarstr, ' = ', outerloopvar)

    loopvarstr="Nhid"
        
    loopvals=[\
    1000, 1300, 1600, 2000, 2500, 3200, 4000, 5000, 6300, 8000, 10000, \
    13000, 16000, 20000, 25000, 32000, 40000, 50000, 63000, 80000, 100000]

    ## annoyingly, we need lists for pcompete
    ## setattr(x,'pcompete[0]',0.1) does not work
    #loopvals2 = loopvals.copy()
   # for i in range(len(loopvals)):
    #    loopvals2[i]=[loopvals[i], 1]  # for pcompete
    #    #loopvals2[i]=[loopvals[i], loopvals[i]] # e.g. for siw0

    parlist=[]
    for loopvar in loopvals:
        param=parameters()
        setattr(param, loopvarstr, loopvar)
        outerloopvar2= [outerloopvar, outerloopvar]
        setattr(param, outerloopvarstr, outerloopvar2)
        param.settingstr = dict_to_string(vars(param), skiplist)
        parlist.append(param)
    #    if (parlist[ip].learnratescaleQ):
    #        parlist[ip].learn_rate[0] /= np.sqrt(parlist[ip].pcompete[0])


    skiplist.append(loopvarstr)
    settingstr = dict_to_string(vars(parlist[0]),skiplist)
    settingstr += '_vs_'+ loopvarstr
    if (Npouter > 1):
        settingstr += '_across_'+ outerloopvarstr
    settingstrwrap=settingstr.replace('_',' ')
    print('bp4.py settings: ', settingstrwrap)
    settingstrwrap = textwrap.fill(settingstrwrap, 80)

    maxcpu= 25
    pool = Pool(maxcpu)
    ################### CALLING NETWORK RUN ====
    outcome = pool.map(runnet, parlist)
    ####################
    pool.close()

    #UGLY
    outcome = np.array(outcome)
    outar = outcome[:,0:13] # version 2: returns Np x 13 array, and prepend 1st col here
    woutar= outcome[:,13:]
    outar   = np.c_[loopvals,outar] # prepend 1st col.
    woutar  = np.c_[loopvals,woutar] # V2  prepend 1st col.
    #np.savetxt('rawdata'+settingstr+'.dat',outar)

    # remove not converged runs
    dellist=[]
    for i in range(outar.shape[0]): 
        if outar[i,3] < parlist[0].accuracy_goal:
            dellist.append(i)
    print('bp4.py non-converging runs : ', dellist)
    loopvals = np.delete(loopvals,dellist,0)
    outar   = np.delete(outar,dellist,0)
    woutar  = np.delete(woutar,dellist,0)
    
    with open('data'+settingstr+'.dat', "ab") as outfile:
        np.savetxt(outfile, outar, fmt='%1.4g',header=settingstr)
        outfile.write(b"\n")

    with open('wdata'+settingstr+'.dat', "ab") as woutfile:
        np.savetxt(woutfile, woutar, fmt='%1.4g',header=settingstr)
        woutfile.write(b"\n")

    # output
    # 0:prepended parameter (p), 1:p actual, 2: iter 3: testerror
    # [L0energy] 4: mlay0   5:mlay1     6:sum   7:mdirect lay0  8:mdirect lay1 
    # [L0energy] 9          10          11      12              13
    
    idx_offset=5
    
    # PLOT
    # iter vs p
    axs[0,0].plot(loopvals, outar[:,2])
    axs[0,0].set_ylim(bottom=0)
    axs[0,0].set( ylabel='iter')
    plt.xscale('log')

    # test and train error, Now gone.
   
    # ineff, 
    axs[1,0].plot(loopvals, outar[:,6] / (outar[:,7]+outar[:,8]))
    axs[1,0].set(xlabel=loopvarstr, ylabel='L0 ineff')

    axs[2,0].plot(loopvals, outar[:,6+idx_offset] / (outar[:,7+idx_offset]+outar[:,8+idx_offset]))
    axs[2,0].set(xlabel=loopvarstr, ylabel='L1 ineff')

    # left-middle column, L0
    legendstr= outerloopvarstr+'='+"{0:.3g}".format(outerloopvar)
    axs[0,1].plot(loopvals, outar[:,6], label=legendstr)
    axs[0,2].set_yscale('log')
    #axs[0,1].set_ylim(bottom=0)
    axs[0,1].set(ylabel='total M0')
    if ( Npouter >1 ):
    	axs[0,1].legend(fontsize='x-small')

    axs[1,1].plot(loopvals, outar[:,4])
    axs[1,1].set(ylabel='M-layer0')

    axs[2,1].plot(loopvals, outar[:,5])
    axs[2,1].set(xlabel=loopvarstr, ylabel='M0-layer1')
    
    # right-middle column, L1 energy
    axs[0,2].plot(loopvals, outar[:,6+idx_offset], label=legendstr)
    #axs[0,2].set_ylim(bottom=0)
    axs[0,2].set_yscale('log')
    axs[0,2].set(ylabel='total M-L1')
    if ( Npouter >1 ):
    	axs[0,2].legend(fontsize='x-small')

    axs[1,2].plot(loopvals, outar[:,4+idx_offset])
    axs[1,2].set(ylabel='M-lay0')

    axs[2,2].plot(loopvals, outar[:,5+idx_offset])
    axs[2,2].set(xlabel=loopvarstr, ylabel='M lay1')
    

    # right column
    #           0,1        2,3  4,5   6,7    8.9    10,11   12,13
    #outw = [wmean[0,1] , wstd, wL1,  Dwmean, Dwstd, DwL1, Dw_persyn]
    axs[0,3].plot(loopvals, woutar[:,3], label = 'Layer 0')
    axs[0,3].plot(loopvals, woutar[:,4], label = 'Layer 1')
    axs[0,3].legend(fontsize='x-small')
    axs[0,3].set(ylabel = 'std(w)')
    axs[0,3].set_ylim(bottom=0)

    axs[1,3].plot(loopvals, woutar[:,4])
    axs[1,3].plot(loopvals, woutar[:,5])
    axs[1,3].set(ylabel='L1(w)')
    axs[1,3].set_ylim(bottom=0)

    axs[2,3].plot(loopvals, woutar[:,12])
    axs[2,3].plot(loopvals, woutar[:,13])
    axs[2,3].set(ylabel='nonsense')

    fig.suptitle(settingstrwrap,fontsize='small')
    fig.show()

fig.savefig("fig"+settingstr+".pdf", bbox_inches='tight')
