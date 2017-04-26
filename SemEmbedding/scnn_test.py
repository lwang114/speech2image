import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from timeit import timeit
from scipy.io import wavfile

# This script used the pre-trained parameters from semanticembed.py
# to build a spectrogram CNN, and test it on a large test dataset,
# which may not be accomodatable in GPU

# Copyright (c) 2006 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""Compute MFCC coefficients.
    
    This module provides functions for computing MFCC (mel-frequency
    cepstral coefficients) as used in the Sphinx speech recognition
    system.
    """

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision: 6390 $"

import numpy, numpy.fft

def mel(f):
    return 2595. * numpy.log10(1. + f / 700.)

def melinv(m):
    return 700. * (numpy.power(10., m / 2595.) - 1.)

class MFCC(object):
    def __init__(self, nfilt=40, ncep=13,
                 lowerf=133.3333, upperf=6855.4976, alpha=0.97,
                 samprate=16000, frate=160, wlen=0.0256,
                 nfft=512):
        # Store parameters
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.ncep = ncep
        self.nfilt = nfilt
        self.frate = frate
        self.fshift = float(samprate) / frate
        
        # Build Hamming window
        self.wlen = int(wlen * samprate)
        self.win = numpy.hamming(self.wlen)
        
        # Prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha
        
        # Build mel filter matrix
        self.filters = numpy.zeros((nfft/2+1,nfilt), 'd')
        dfreq = float(samprate) / nfft
        if upperf > samprate/2:
            raise(Exception,
                  "Upper frequency %f exceeds Nyquist %f" % (upperf, samprate/2))
        melmax = mel(upperf)
        melmin = mel(lowerf)
        dmelbw = (melmax - melmin) / (nfilt + 1)
        # Filter edges, in Hz
        filt_edge = melinv(melmin + dmelbw * numpy.arange(nfilt + 2, dtype='d'))
        
        for whichfilt in range(0, nfilt):
            # Filter triangles, in DFT points
            leftfr = round(filt_edge[whichfilt] / dfreq)
            centerfr = round(filt_edge[whichfilt + 1] / dfreq)
            rightfr = round(filt_edge[whichfilt + 2] / dfreq)
            # For some reason this is calculated in Hz, though I think
            # it doesn't really matter
            fwidth = (rightfr - leftfr) * dfreq
            height = 2. / fwidth
            
            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.filters[freq,whichfilt] = (freq - leftfr) * leftslope
                freq = freq + 1
            if freq == centerfr: # This is always true
                self.filters[freq,whichfilt] = height
                freq = freq + 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.filters[freq,whichfilt] = (freq - rightfr) * rightslope
                freq = freq + 1
                #             print("Filter %d: left %d=%f center %d=%f right %d=%f width %d" %
                #                   (whichfilt,
                #                   leftfr, leftfr*dfreq,
                #                   centerfr, centerfr*dfreq,
                #                   rightfr, rightfr*dfreq,
                #                   freq - leftfr))
                #             print self.filters[leftfr:rightfr,whichfilt]
                # Build DCT matrix
                self.s2dct = s2dctmat(nfilt, ncep, 1./nfilt)
                self.dct = dctmat(nfilt, ncep, numpy.pi/nfilt)

    def sig2s2mfc(self, sig):
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = numpy.zeros((nfr, self.ncep), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = numpy.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2s2mfc(frame)
            fr = fr + 1
        return mfcc

    def sig2logspec(self, sig):
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = numpy.zeros((nfr, self.nfilt), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = numpy.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2logspec(frame)
            fr = fr + 1
        return mfcc

    def pre_emphasis(self, frame):
        # FIXME: Do this with matrix multiplication
        outfr = numpy.empty(len(frame), 'd')
        outfr[0] = frame[0] - self.alpha * self.prior
        for i in range(1,len(frame)):
            outfr[i] = frame[i] - self.alpha * frame[i-1]
        self.prior = frame[-1]
        return outfr

    def frame2logspec(self, frame):
        frame = self.pre_emphasis(frame) * self.win
        fft = numpy.fft.rfft(frame, self.nfft)
        # Square of absolute value
        power = fft.real * fft.real + fft.imag * fft.imag
        return numpy.log(numpy.dot(power, self.filters).clip(1e-5,numpy.inf))
    
    def frame2s2mfc(self, frame):
        logspec = self.frame2logspec(frame)
        return numpy.dot(logspec, self.s2dct.T) / self.nfilt

def s2dctmat(nfilt,ncep,freqstep):
    """Return the 'legacy' not-quite-DCT matrix used by Sphinx"""
    melcos = numpy.empty((ncep, nfilt), 'double')
    for i in range(0,ncep):
        freq = numpy.pi * float(i) / nfilt
        melcos[i] = numpy.cos(freq * numpy.arange(0.5, float(nfilt)+0.5, 1.0, 'double'))
    melcos[:,0] = melcos[:,0] * 0.5
    return melcos

def logspec2s2mfc(logspec, ncep=13):
    """Convert log-power-spectrum bins to MFCC using the 'legacy'
        Sphinx transform"""
    nframes, nfilt = logspec.shape
    melcos = s2dctmat(nfilt, ncep, 1./nfilt)
    return numpy.dot(logspec, melcos.T) / nfilt

def dctmat(N,K,freqstep,orthogonalize=True):
    """Return the orthogonal DCT-II/DCT-III matrix of size NxK.
        For computing or inverting MFCCs, N is the number of
        log-power-spectrum bins while K is the number of cepstra."""
    cosmat = numpy.zeros((N, K), 'double')
    for n in range(0,N):
        for k in range(0, K):
            cosmat[n,k] = numpy.cos(freqstep * (n + 0.5) * k)
    if orthogonalize:
        cosmat[:,0] = cosmat[:,0] * 1./numpy.sqrt(2)
    return cosmat

def dct(input, K=13):
    """Convert log-power-spectrum to MFCC using the orthogonal DCT-II"""
    nframes, N = input.shape
    freqstep = numpy.pi / N
    cosmat = dctmat(N,K,freqstep)
    return numpy.dot(input, cosmat) * numpy.sqrt(2.0 / N)

def dct2(input, K=13):
    """Convert log-power-spectrum to MFCC using the normalized DCT-II"""
    nframes, N = input.shape
    freqstep = numpy.pi / N
    cosmat = dctmat(N,K,freqstep,False)
    return numpy.dot(input, cosmat) * (2.0 / N)

def idct(input, K=40):
    """Convert MFCC to log-power-spectrum using the orthogonal DCT-III"""
    nframes, N = input.shape
    freqstep = numpy.pi / K
    cosmat = dctmat(K,N,freqstep).T
    return numpy.dot(input, cosmat) * numpy.sqrt(2.0 / K)

def dct3(input, K=40):
    """Convert MFCC to log-power-spectrum using the unnormalized DCT-III"""
    nframes, N = input.shape
    freqstep = numpy.pi / K
    cosmat = dctmat(K,N,freqstep,False)
    cosmat[:,0] = cosmat[:,0] * 0.5
    return numpy.dot(input, cosmat.T)

def loadtest(ntx):
    infofile = '../data/flickr_audio/wav2capt.txt'
    dir_sp = '../data/flickr_audio/wavs/'
    dir_penult = '../data/vgg_flickr8k_nnet_penults/'
    
    # If the file does not exist in the path above, default the user has put the data inside the current directory
    if not os.path.isfile(infofile):
        infofile = 'flickr_audio/wav2capt.txt'

    mfcc = MFCC()
    captions_tx = []
    im_tx = []
    Leq = 1024
    with open(infofile, 'r') as f:
        for j in range(ntx):
            # Load the image names and the image captions, break the captions into words and store in a list
            cur_info = f.readline()
            cur_info_parts = cur_info.rstrip().split()
            sp_name = cur_info_parts[0]
            if os.path.isfile(dir_sp+sp_name):
                caption_info = wavfile.read(dir_sp+sp_name)
                caption_time = caption_info[1]
                caption = mfcc.sig2logspec(caption_time)
            else:
                caption_info = wavfile.read(sp_name)
                caption_time = caption_info[1]
                caption = mfcc.sig2logspec(caption_time)

            # Transpose the data
            if caption.shape[0] > caption.shape[1]:
                caption = np.transpose(caption)
            # Equalize the length
            if caption.shape[1] < Leq:
                nframes = caption.shape[1]
                nmf = caption.shape[0]
                caption_new = np.zeros((nmf, Leq))
                #print('274:', nframes, (Leq-nframes)/2)
                caption_new[:, (Leq-nframes)/2:(Leq-nframes)/2+nframes] = caption
            else:
                if caption.shape[1] > Leq:
                    nframes = caption.shape[1]
                    nmf = caption.shape[0]
                    caption_new = np.zeros((nmf, Leq))
                    caption_new = caption[:, (nframes-Leq)/2:(nframes-Leq)/2+Leq]
                    captions_tx.append(caption_new)
            # Remove the .jpg# at the end of the file to the format of vgg16 feature file
            im_name_raw = cur_info_parts[1]
            im_name_parts = im_name_raw.split('.')
            #len_im_name = len(im_name_parts[0])
            # Remove the caption number
            im_name = im_name_parts[0]
            # Load the softmax activations of the images, store them into an array
            # If the file is not in a folder called data, then by default it is in the current dir
            if os.path.isfile(dir_penult+im_name+'.npz'):
                data = np.load(dir_penult+im_name+'.npz')
            else:
                data = np.load(im_name+'.npz')
            cur_penult = data['arr_0']
            im_tx.append(cur_penult)
            if j % 10:
                print('Finish loading', 100*j/ntx, 'percent of test data')
    captions_tx = np.array(captions_tx)
    im_tx = np.array(im_tx)
    np.savez('captions_test.npz', captions_tx)
    np.savez('images_test.npz', im_tx)
    return captions_tx, im_tx



def scnn_test(ntx):
    # Load test data
    sp_test = 'captions.npz'#'captions_test.npz'
    im_test = 'images.npz'#'images_test.npz'
    
    if os.path.isfile(sp_test):
        data = np.load(sp_test)
        #if data['arr_0'].shape[0] == ntx:
        X_test = data['arr_0'][0:ntx]
        data_im = np.load(im_test)
        Z_test_vgg = data_im['arr_0'][0:ntx]
        #else:
        #X_test, Z_test_vgg = loadtest(ntx)
    else:
        X_test, Z_test_vgg = loadtest(ntx)
    
    nmf = X_test[0].shape[0]
    nframes = X_test[0].shape[1]
    print('number of data in X_test and Z_test_vgg:', X_test.shape, Z_test_vgg.shape)
    #print('the current frame size and number of mel frequencies are:', nframes, nmf)
    X_test_4d = np.reshape(X_test[0:ntx], [ntx, 1, nframes, nmf])

    nlabel = 61
    N = [4, 24, 24]
    J = [nmf, 64, 512, 1024, nlabel]
    D = [nframes, nframes/2, nframes/4, nframes/4]
    npenult_vgg = 4096
    nembed = 1024

    w_in = tf.placeholder(tf.float32, shape=[1, N[0]+1, J[0], J[1]])
    b_in = tf.placeholder(tf.float32, shape=[J[1]])

    w_hidden1 = tf.placeholder(tf.float32, shape=[1, N[1]+1, J[1], J[2]])
    b_hidden1 = tf.placeholder(tf.float32, shape=[J[2]])

    w_hidden2 = tf.placeholder(tf.float32, shape=[1, N[2]+1, J[2], J[3]])
    b_hidden2 = tf.placeholder(tf.float32, shape=[J[3]])

    w_out = tf.placeholder(tf.float32, shape=[J[3], J[4]])
    b_out = tf.placeholder(tf.float32, shape=[J[4]])

    X_in = tf.placeholder(tf.float32, shape=[None, 1, nframes, nmf])
    X_mean = tf.reduce_mean(X_in)
    X_zm = X_in - X_mean

    a1_conv = tf.nn.conv2d(X_zm, w_in, strides=[1, 1, 1, 1], padding='SAME') + b_in
    # Max pooling with vertical stride 1 and horizontal stride 2
    a1_pool = tf.nn.max_pool(a1_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h1 = tf.nn.relu(a1_pool)
    
    a2_conv = tf.nn.conv2d(h1, w_hidden1, strides=[1, 1, 1, 1], padding='SAME') + b_hidden1
    a2_pool = tf.nn.max_pool(a2_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h2 = tf.nn.relu(a2_pool)

    a3_conv = tf.nn.conv2d(h2, w_hidden2, strides=[1, 1, 1, 1], padding='SAME') + b_hidden2
    h3 = tf.nn.relu(a3_conv)
    
    # Penultimate layer
    h4 = tf.nn.max_pool(h3, ksize=[1, 1, D[3], 1], strides=[1, 1, 1, 1], padding='VALID')
    h4_re = tf.reshape(h4, [-1, J[3]])
    # L2 normalization
    h4_ren = tf.nn.l2_normalize(h4_re, dim=1)
    
    a_out = tf.matmul(h4_ren, w_out) + b_out
    Y_pred = tf.nn.softmax(a_out)

    # Map the penultimate vgg vector to the semantic space
    Z_penult_vgg = tf.placeholder(tf.float32, shape=[None, npenult_vgg])
    w_embed = tf.placeholder(tf.float32, shape=[npenult_vgg, nembed])
    b_embed = tf.placeholder(tf.float32, shape=[nembed])
    Z_embed_vgg = tf.matmul(Z_penult_vgg, w_embed) + b_embed
    
    s_a = tf.matmul(h4_ren, tf.transpose(Z_embed_vgg))
    s = tf.nn.relu(s_a)
    # assume the number of features is nbatch
    s_p = tf.diag_part(s)
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Load network parameters
    data = np.load('scnn_pmtrs.npz')
    pmtrs = data['arr_0']
    _w_in = pmtrs[0]
    _b_in = pmtrs[1]
    _w_hidden1 = pmtrs[2]
    _b_hidden1 = pmtrs[3]
    _w_hidden2 = pmtrs[4]
    _b_hidden2 = pmtrs[5]
    _w_out = pmtrs[6]
    _b_out = pmtrs[7]

    data = np.load('vgg_pmtrs.npz')
    pmtrs_vgg = data['arr_0']
    _w_embed = pmtrs_vgg[0]
    _b_embed = pmtrs_vgg[1]

    batch_size = 128
    nbatch = int(ntx/batch_size)
    _Z_embed_sp = np.zeros((ntx, nembed))
    _Z_embed_vgg = np.zeros((ntx, nembed))
    for k in range(nbatch):
        X_batch = X_test_4d[batch_size*k:batch_size*(k+1)]
        Z_batch = Z_test_vgg[batch_size*k:batch_size*(k+1)]
        Z_curr_sp = sess.run(h4_ren, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
        Z_curr_vgg = sess.run(Z_embed_vgg, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
        _Z_embed_sp[batch_size*k:batch_size*(k+1)] = Z_curr_sp
        _Z_embed_vgg[batch_size*k:batch_size*(k+1)] = Z_cur_vgg

    similarity = _Z_embed_sp * np.transpose(_Z_embed_vgg)
    #X_tx_4d = X_stack_tx.reshape([ntx*(nframes-2*nreduce), 1, nwin, nmf])
    #test_accuracy = sess.run(accuracy, feed_dict={X_in:X_tx_4d, Z_in:Z_tx})
    ntop = 10
    top_indices = np.zeros((ntop, ntx))
    for k in range(ntop):
        # Find the most similar image feature of the speech feature on the penultimate feature space
        cur_top_idx = np.argmax(similarity, axis=1)
        top_indices[k] = cur_top_idx
        # To leave out the top values that have been determined and the find the top values for the rest of the indices
        similarity[cur_top_idx] = -1;
    # Find if the image with the matching index has the highest similarity score
    #dev = abs(np.transpose(np.transpose(top10_indices) - np.linspace(0, ntr-1, ntr)))
    dev = abs(top_indices - np.linspace(0, ntx-1, ntx))
    min_dev = np.amin(dev, axis=0)
    # Count the number of correct matching by counting the number of 0s in dev
    test_accuracy = np.mean(min_dev == 0)
    print('Test accuracy is: ', str(test_accuracy))

ntx = int(sys.argv[1])
scnn_test(ntx)
