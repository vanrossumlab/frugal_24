# version 2022-3-2
import numpy as np
import copy
import random
from scipy.stats import bernoulli

#epsilon=1e-10 # small number to prevent log divergence in xtropy
global_a_lru = 0.1 # parameter for Leaky ReLu

def vars_to_string(**vars): # utility to create string with parameter value
    sout = ''
    for (name, val) in vars.items():
        sout += "%s_%s" % (name, repr(val))
    return sout

def transform_labels_into_one_hot(labels, N_distinct_labels):
    q_xtropy=0  # when you want to vary this, add an argument.
    lr = np.arange( N_distinct_labels )
    labels_one_hot = (lr==labels).astype(np.float32)
    labels_one_hot[labels_one_hot == 1] = 1-q_xtropy
    labels_one_hot[labels_one_hot == 0] = q_xtropy/(N_distinct_labels-1)
    return labels_one_hot

# to implement:
#https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-input
def sigmoid(x, outputlayerQ, xtropyQ):
    if outputlayerQ:
        if xtropyQ: # use softmax for output layer 
            y=np.exp(x)
            return y/sum(y)
        else:         # if MSE use linear output
            return x
    else:
        return 1/(1 + np.exp(-x))

def dsigmoid(y): # if y is already known...
    return y*(1.0-y)

def reluf(x, outputlayerQ, xtropyQ): # ReLU units
    if outputlayerQ:
        if xtropyQ: 
            y=np.exp(np.minimum(200,x)) # prevents num overflow
            return y/sum(y)
        else:
            return x
    else:
        return x*(x>0)

def dreluf(y):
    return 1.0*(y>0).astype('float32')

def eluf(x, outputlayerQ, xtropyQ): # Exp. ReLU units
    if outputlayerQ:
        if xtropyQ: 
            y=np.exp(np.minimum(200,x))
            return y/sum(y)
        else:
            return x
    else:
        return x*(x>0)+(-1+np.exp(x))*(x<=0)

def deluf(y):
    return (1.0*(y>0)+(y+1)*(y<=0)).astype('float32')

def lruf(x, outputlayerQ, xtropyQ): # Leaky ReLU units
    if outputlayerQ:
        if xtropyQ: 
            y=np.exp(np.minimum(200,x)) 
            return y/sum(y)
        else:      
            return x
    else:
        return x*(x>0)+global_a_lru*x*(x<=0) 
    #could be sped up?

#https://stackoverflow.com/questions/43233535/explicitly-define-datatype-in-python-function
def dlruf(y):
    return (global_a_lru+(1.0-global_a_lru)*(y>0)).astype('float32')
 
def make_mask(p, sizein):
    # NOte, if we have just a single mask, a regular mask would suffice.
    nel = np.prod(sizein)
    out = np.zeros(nel)
    k   = round(p*nel)
    out[0:k]=1
    out = np.random.permutation(out)
    out = out.reshape(sizein)
    return out

def thr_array(filtar, testar, fraction, invQ=False, perNeuronQ=False):
    # Select elements in filtarry for which corresponding testar elements are in top fraction
    # other elements are set to zero. Typically, filtar = dw, testar =abs(dw_proposed)
    # Output: filtar with zeros whenever not in fraction
    # Input: filtar, testar:
    # if fraction = 1 => return all elements
    # if invQ: => select bottom values.

    if testar.shape != filtar.shape :
        print("testar.shape != filtar.shape :" , testar.shape, filtar.shape)
        raise
    if invQ:    
        testar  = -testar
    if not(perNeuronQ):    
        k       = round( (1-fraction)*(testar.size-1))
        outar   = filtar
        # new method should be quicker and more robust to equal entries.
        # axis=None: use flattenen matrix

        ####outar[np.argpartition(testar, k-1, axis=None)[:k] ]=0.0
        # BROKEN : index 32645 is out of bounds for axis 0 with size 785
        # need to unflatten indices...

        # old method:
        thr     = np.partition(testar, k, axis=None)[k]
        outar   = np.where(testar >= thr, filtar, 0)
    else: # per Neuron
        # the w and dw matrix are INPUT x OUTPUT
        # we select on the largest updates for each post, i.e. per column 
        
        k       = round( (1-fraction)*(testar.shape[0]-1) )
        k       = min(k, testar.shape[0]-1)
        idxs    = np.argpartition(testar, k, axis=0)[:k,:] 
        outar   = filtar
        np.put_along_axis(outar, idxs, 0.0, axis=0)
        
        #k       = round( (1-fraction)*testar.shape[0] )
        #thrs    = np.partition(testar, k, axis=0)[k,:] 
        #outar   = np.where(testar >= thrs, filtar, 0)
    return outar
    
def runnet(param):
    if param.version !=2:
        print("Incompatible version. Exiting")
        raise
    writewQ = False # save file with with stats?
    
    # READ DATASET
    image_size = 28
    N_distinct_labels = 10
    image_pixels = image_size*image_size

    if(param.smalldatasetQ):
        train_data=np.fromfile("mnist_data/mnist_train_imgs1000.dat").reshape((1000,image_pixels))
        train_labels=np.fromfile("mnist_data/mnist_train_lbls1000.dat",int).reshape((1000,1))
        test_data=np.fromfile("mnist_data/mnist_test_imgs1000.dat").reshape((1000,image_pixels))
        test_labels=np.fromfile("mnist_data/mnist_test_lbls1000.dat",int).reshape((1000,1))
    else:
        train_data=np.fromfile("mnist_data/mnist_train_imgs.dat").reshape((60000,image_pixels))
        train_labels=np.fromfile("mnist_data/mnist_train_lbls.dat",int).reshape((60000,1))
        test_data=np.fromfile("mnist_data/mnist_test_imgs.dat").reshape((10000,image_pixels))
        test_labels=np.fromfile("mnist_data/mnist_test_lbls.dat",int).reshape((10000,1))
    
    if param.biasQ:
        train_data = np.concatenate((train_data,np.ones((len(train_data),1))),axis=1)
        test_data =  np.concatenate((test_data,np.ones((len(test_data),1))), axis=1)

    train_data = train_data -0.01 
    # hack to compensate for the fact that data =0.01..0.99. 
    test_data = test_data -0.01

    train_goal = transform_labels_into_one_hot(train_labels, N_distinct_labels)
    train_labels=train_labels.T[0]  # from 2D 60000x1 arrays to 1D 
    test_labels=test_labels.T[0]
    if param.zeromeandataQ:
        train_data = train_data - np.mean(train_data) 
        test_data =  test_data  - np.mean(test_data)
    
    # add a bit of noise to make thr_array selection unique.
    siin=0.001
    train_data = train_data+np.random.normal(0,siin,train_data.shape)
    test_data = test_data+np.random.normal(0,siin,test_data.shape)

    datatype    = 'float32'
    train_data  = train_data.astype(datatype)
    test_data   = test_data.astype(datatype)
    npat = len(train_data) 
    Nin= np.shape(train_data)[1]
    if param.gfun== "relu":
        gfun   = reluf
        gprime = dreluf
    elif param.gfun== "elu":
        gfun   = eluf
        gprime = deluf
    elif param.gfun== "lru":
        gfun   = lruf
        gprime = dlruf
    elif param.gfun== "sigmoid":   
        gfun   = sigmoid
        gprime = dsigmoid
    else:   
        print("runnet.py: no such activation function!")
        raise

    nlayer      = 3 # depth of network. if single hidden: nlayer =3
    outlayer    = nlayer-2 # for easy indexing output layer 
    
#### Setup network.   
    # y[0]  >  w[0] >  y[1]  >  w[1]  > y[2]
    # backprop, see HKP, but there w[2] is the matrix connecting y[1] to y[2]
    # w has dimensions #inputs x # outputs
    w       = [] # synaptic weights, list of matrices
    dw      = [] # updates
    bperror = [] # backprop error
    staticlayerQ = []
    
    neuronmask = [] # plasticity masks
    layermask = []
    
    Nout=N_distinct_labels 
    if param.compete_algol == 2020 or param.compete_algol == 104: 
        param.pcompete[1] = np.minimum( param.pcompete[0]*78.4, 1.0)        
    
    # creates network, Nin, Nhid, Nhid,..., Nout
    # should I increase network for bias? Currently, bias units are clamped.
    for ilayer in range(nlayer-1): # 0,1 there is one less matrix than layers.
        n_in    = param.Nhid if ilayer > 0 else Nin
        n_out   = param.Nhid if ilayer < outlayer else Nout
        wtmp    = np.random.normal(0.0,param.si_w0[ilayer],(n_in,n_out))
        w.append(wtmp.astype(datatype))
        dw.append(wtmp.astype(datatype))
        tmp     = np.zeros(n_out)
        bperror.append(tmp.astype(datatype))
        if param.compete_algol == 4 : # fixed mask
            #neuronmask.append(bernoulli.rvs(param.pcompete[ilayer], size = tmp.shape))
            neuronmask.append( make_mask(param.pcompete[ilayer], tmp.shape).astype(datatype))
        elif param.compete_algol == 6 : # multiple fixed masks, one per digit
            neuronmask.append([])
            for jlabel in range(N_distinct_labels):
                #neuronmask[ilayer].append(bernoulli.rvs(param.pcompete[ilayer], size = tmp.shape).astype(datatype))
                neuronmask[ilayer].append(make_mask(param.pcompete[ilayer],tmp.shape).astype(datatype))
        elif param.compete_algol == 10: # alternating mask
            #nmasks = 1 # should be param. enrtry
            neuronmask.append([])
            for i in range(param.nmasks):
                neuronmask[ilayer].append(bernoulli.rvs(param.pcompete[ilayer], size = tmp.shape))
        staticlayerQ.append(False)
        if param.compete_algol == 104 : # fixed mask on layer basis
            #layermask.append(bernoulli.rvs(param.pcompete[ilayer], size = w.shape))
            layermask.append( make_mask(param.pcompete[ilayer], w[ilayer].shape).astype(datatype))
            #print(' mask ', ilayer, param.pcompete[ilayer], layermask[ilayer].mean())
        if param.compete_algol == 105 : # fixed mask with subset of plastic in-out
            nplast= round(param.pcompete[0]*param.Nhid)        # note param.compete[1] is ignored
            mask= np.zeros(w[ilayer].shape)
            if ilayer==0:
                mask[:,:nplast]=1
            else:
                mask[:nplast,:]=1
            layermask.append( mask.astype(datatype) )
        if param.compete_algol == 106 : # fixed mask with subset of plastic in-out
            nplast= param.k_algolg 
            mask= np.zeros(w[ilayer].shape)
            if ilayer==0:
                mask[:,:nplast]=1
            else:
                mask[:nplast,:]=1
            layermask.append( mask.astype(datatype) )    
            
    winit = copy.deepcopy(w)

    y = [] # activities, list of vectors
    for ilayer in range(nlayer): #0,1,2
        n = param.Nhid if ilayer > 1 else Nin
        n = n if ilayer < nlayer else Nout
        ytmp = np.zeros(n).astype(datatype)
        y.append(ytmp.astype(datatype))
        
    # START
    mL1 = np.zeros(nlayer-1) # metabolic energy, L1 norm
    mL0 = np.zeros(nlayer-1) # metabolic energy, L0 norm  STILL TO WRITE OUTPUT
    fracplast = np.zeros(nlayer-1)
    fracplast_cum = np.zeros(nlayer-1)
    fracactive = np.zeros(nlayer)
    iter = 0; mse_error= 0; xtropy= 0 ; train_error= 0
    results_vs_iter=[]; results_vs_acc=[]
    dmlist =[[],[]]
    doneQ = False
    
    if (param.compete_algol == 7):  # static hidden layer. Cache y[1]
        # need to calculate y[hidden] only once
        # Can't have Static layer after Non-static
        staticlayerQ[0] = True
        ystored=np.zeros([len(train_data), 2, param.Nhid])
        for ipat in range(len(train_data)):
            y[0] = train_data[ipat]
            for ilayer in range(nlayer-2): #forward pass 0
                if (param.biasQ & ilayer >0):
                    y[ilayer][0]=1.0
                ystored[ipat, ilayer+1,:] = gfun(np.dot(y[ilayer],w[ilayer]), ilayer==outlayer, param.xtropyQ)

    for iepoch in range(param.maxepochs):
        if (doneQ):
            print("Done at epoch = ", iepoch)
            break
        else:
            print(iepoch)

        for _ in range(len(train_data)):
            if (doneQ): break
            iter +=1
            
            ipat    = random.randrange(len(train_data)) #pick random pattern
            y[0]    = train_data[ipat] 
            ilabel  = train_labels[ipat] # output label, used for output dep. methods
            for ilayer in range(nlayer-1): #forward pass 0,1
                if (staticlayerQ[ilayer]):
                    y[ilayer+1] = ystored[ipat,ilayer+1,:]
                else:
                    if (param.biasQ & ilayer >0):
                        y[ilayer][0]=1.0
                    y[ilayer+1] = gfun(np.dot(y[ilayer],w[ilayer]), ilayer==outlayer, param.xtropyQ)

            # output error (assuming linear output and MSE, or softmax xtropy)   
            bperror[outlayer]= (train_goal[ipat] - y[-1]).astype(datatype)
            
            # xtropy      += np.sum(np.log(1-y[-1]))+np.log(y[-1][train_labels[ipat]])-np.log(1+epsilon-y[-1][train_labels[ipat]]) if param.xtropyQ else 0 # not really used
            #mse_error   += sum(bperror[outlayer]**2)
            #print('mse_error = ', mse_error)
            train_error += int(y[-1].argmax() != train_labels[ipat] ) # only used for printing         
           
            # calculate backprop error;
            for ilayer in reversed(range(nlayer-2)): # if one hidden: ilayer= 0 only
                if (staticlayerQ[ilayer]):
                    continue
                bperror[ilayer] =  gprime(y[ilayer+1])* np.dot(w[ilayer+1],bperror[ilayer+1])
                
            # calculate weight update 
            for ilayer in reversed(range(nlayer-1)): # for one hidden: = 1, 0
                if (staticlayerQ[ilayer]):
                    continue
                                                        
                # I. First we perform selection on bperror. This will propagate back.                                        
                if param.compete_algol  in [0,7,20,2020,21,22,23,40,41,104,105,106]: # plain backprop, we deal with this below
                    pass
                
                elif param.compete_algol == 1: # random, per neuron, could normalize learn_rate
                    bperror[ilayer] *= bernoulli.rvs(param.pcompete[ilayer], size= bperror[ilayer].shape)
                
                elif param.compete_algol == 2: # Obsolete, removed
                    raise
                
                elif param.compete_algol == 4: # 
                    if (param.pcompete[ilayer] != 1): # skip over case when p=1                
                        bperror[ilayer] *= neuronmask[ilayer]
                
                elif param.compete_algol == 5: # 
                    error # TODO  dw*= mask
                
                elif param.compete_algol == 6: # 
                    if (param.pcompete[ilayer] != 1): # skip over p=1
                        bperror[ilayer] *= neuronmask[ilayer][ilabel]    
                
                elif param.compete_algol == 10: #alternating masks
                    if (param.pcompete[ilayer] != 1): 
                        bperror[ilayer] *= neuronmask[ilayer][iter % param.nmasks]     
                        
                  # algol 30-39:  mask bp-error 
                elif param.compete_algol == 30: 
                    if param.pcompete[ilayer] !=1 :
                        bperror[ilayer] = thr_array(bperror[ilayer], abs(bperror[ilayer]), param.pcompete[ilayer])

                elif param.compete_algol == 31: # neuronmask dependent on abs(bperror)
                    if param.pcompete[ilayer] !=1 :
                        bperror[ilayer] = thr_array(bperror[ilayer], abs(bperror[ilayer]), param.pcompete[ilayer],1)
                
                elif param.compete_algol == 32: # dependent on abs(y)
                    if param.pcompete !=1 :
                        bperror[ilayer] = thr_array(bperror[ilayer], abs(y[ilayer+1]), param.pcompete[ilayer])

                elif param.compete_algol == 33: # dependent on abs(y)
                    if param.pcompete !=1 :
                        bperror[ilayer] = thr_array(bperror[ilayer], abs(y[ilayer+1]), param.pcompete[ilayer],1)
                else:
                    error
            
                dw[ilayer]  = param.learn_rate[ilayer]*(np.outer(y[ilayer], bperror[ilayer]) - param.lam2reg*w[ilayer])

                # II. Below we deal we selection in dw. This will not backpropagate 
                # filter updates on per layer basis
                # !! TODO: regularization not included
                if param.compete_algol == 20 or param.compete_algol == 2020: # only largest dw (pos and neg) get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(dw[ilayer]), param.pcompete[ilayer])
                
                if param.compete_algol == 2000: # only largest dw (pos and neg) get expressed
                # but also limited to algol_k hidden units. 
                # other are set to zero.
                    if param.algol_k > param.Nhid:
                        print('Warning: param.algol_k > param.Nhid. Setting it to Nhid')
                        param.algol_k = param.Nhid
                    if ilayer ==0:
                        dw[ilayer][:,param.algol_k:] = 0.0;
                    else:
                        dw[ilayer][param.algol_k:,:] = 0.0;
                        
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(dw[ilayer]), param.pcompete[ilayer])        
                elif param.compete_algol == 21: # only largest -w- (pos and neg) get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(w[ilayer]), param.pcompete[ilayer])

                elif param.compete_algol == 22: # only smallest dw (pos and neg) get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(dw[ilayer]), param.pcompete[ilayer], True)

                elif param.compete_algol == 23: # only smallest w (pos and neg) get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(w[ilayer]), param.pcompete[ilayer], True)
                                       
                # filter updates on per neuron basis
                elif param.compete_algol == 40: # only largest dw (pos and neg) PER NEURON get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(dw[ilayer]), param.pcompete[ilayer], False, True)
                
                elif param.compete_algol == 41: # only largest w (pos and neg) PER NEURON get expressed
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] = thr_array(dw[ilayer], abs(w[ilayer]), param.pcompete[ilayer], False, True)
                                
                elif param.compete_algol == 104: # fixed mask per layer 
                    if param.pcompete[ilayer] !=1 :
                        dw[ilayer] *= layermask[ilayer]
                elif param.compete_algol == 105 or param.compete_algol == 106: # fixed mask per layer 
                    dw[ilayer] *= layermask[ilayer]        
                        
                w[ilayer]  += dw[ilayer]
                dm = abs(dw[ilayer]).sum()
                mL1[ilayer] += dm
                nonzeros = np.count_nonzero(dw[ilayer] != 0 )
                mL0[ilayer] += nonzeros
                fracplast[ilayer] += nonzeros/dw[ilayer].size
                fracplast_cum[ilayer] += nonzeros/dw[ilayer].size
                if writewQ:
                    # some stats weight updates over time.
                    # could also be of interest to track stats of y.
                    # used to rely on nplast[ilayer], but now uses nonzeros
                    # note, need to compensate std for non-plastic neurons.
                    
                    frac=nonzeros/dw[ilayer].size
                    si_nz = dw[ilayer].std/f - (dw[ilayer].mean)**2*(1-f)/f^2
                    dmlist[ilayer].append( [iter/npat, dm/nonzeros, si_nz, w[ilayer].std()] )
                       
            # end ilayer
            
            #        
            # performance on test set, and other stats
            #
            if (iter % param.testinterval == 0):
                train_error /= param.testinterval
                xtropy = 0

                fracplast /= param.testinterval
                test_error= 0.0
                for ipat in range(len(test_data)):
                    y[0] = test_data[ipat]
                    for ilayer in range(nlayer-1): # this is a copy of above code
                        if (param.biasQ & ilayer >0):
                            y[ilayer][0] = 1.0
                        y[ilayer+1] = gfun(np.dot(y[ilayer],w[ilayer]),ilayer==outlayer,param.xtropyQ)
                    test_error += int(y[-1].argmax() != test_labels[ipat])
                test_error = test_error/ len(test_data)    
                
                #if (iter % len(train_data)==0):
                #print("\n "train_error = ", train_error, " test_error = ", test_error)                                
                
                # Calculate energy to directly set weight (tortuosity)
                m0direct=np.zeros(nlayer-1)
                m1direct=np.zeros(nlayer-1)
                for ilayer in range(nlayer-1): 
                    m0direct[ilayer] = np.count_nonzero(w[ilayer]-winit[ilayer])
                    m1direct[ilayer] = abs(w[ilayer]-winit[ilayer]).sum()
                
                # outputentry is overwritten every test
                # In version 2 we return both M0 and M1 energies
                outputentry= [param.pcompete[0], iter/npat, 1.0-test_error, \
                    mL0[0], mL0[1], mL0[0]+mL0[1], m0direct[0], m0direct[1], \
                    mL1[0], mL1[1], mL1[0]+mL1[1], m1direct[0], m1direct[1] ]
                
                #not used anymore
                #results_vs_iter.append(outputentry)
                #results_vs_acc.append([1.0 - test_error, mL1.sum(), mdirect.sum(), mL1[0],mdirect[0], mL1[1], mdirect[1], iter ]) # add ratios?
                
                mse_error= 0; train_error= 0; xtropy= 0
                doneQ = (test_error < 1 - param.accuracy_goal)
                
                # check for NaN
                if ( np.isnan(mL1).any() or np.isnan(w[0]).any() ):
                    print("NaN detected, quitting run")
                    doneQ = True
                # TODO : check for unwanted dtype promotion

                fracplast = 0*fracplast # reset

            # end ipat
        #end iepoch
    
    # Done. 
    
    # collect weight stats. Norm of final weights, Amount of change.
    # todo Euclidean norm /Frobenius
    # Dw stands for total change over learning, dw for per sample change  
    
    Dw=[];Dw_persyn=[]
    wmean=[]; wstd=[]; wL1=[]
    Dwmean=[]; Dwstd=[]; DwL1=[]
    fracplast_cum2= []

    filtQ = False
    for ilayer in range(nlayer-1): #0,1
        if filtQ: #filter out static synapses
            staticlist = np.where(neuronmask[ilayer] == 0)
            w[ilayer]  = np.delete(w[ilayer],staticlist, axis =1)
            winit[ilayer]= np.delete(winit[ilayer], staticlist, axis=1)
            dw[ilayer]  = np.delete(dw[ilayer],staticlist, axis =1)

        Dw.append( w[ilayer]-winit[ilayer] )
        # for small initial weights Dw \approx w

        # careful Dw can be zero because not plastic, or because y<0.
        #Dwnozero = (Dw[ilayer])[np.where(Dw != 0)] # 1D array 
        #nsynplast = sum(neuronmask[ilayer])* (w[ilayer]).shape[0]

        # note, mean and std are across both axes
        Dw_persyn.append(   (Dw[ilayer]).mean()) # /iter NONSENSE
        wmean.append(       w[ilayer].mean())
        Dwmean.append(      Dw[ilayer].mean())
        
        wL1.append(        abs(w[ilayer]).mean())
        DwL1.append(       abs(Dw[ilayer]).mean())
        
        wstd.append(       w[ilayer].std())
        Dwstd.append(      Dw[ilayer].std())
        fracplast_cum2.append(fracplast_cum[ilayer]/iter)

        if writewQ: 
            dmfilename= 'dm' + str(ilayer)+param.settingstr+'.dat'
            np.savetxt(dmfilename, dmlist[ilayer])
            
            dmfinalfilename = 'dmfinal' + str(ilayer)+param.settingstr+'.dat'
            out= np.transpose([w[ilayer].reshape(-1) ,dw[ilayer].reshape(-1)  ] )
            np.savetxt( dmfinalfilename,out )
    
    outw = wmean + wstd + wL1+   Dwmean + Dwstd + DwL1 + Dw_persyn + fracplast_cum2

   # print('runnet.py. Returning outw: ',outw)

    return outputentry + outw # concats lists.
