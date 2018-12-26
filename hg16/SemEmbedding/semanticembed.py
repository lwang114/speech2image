import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from timeit import timeit
#from IPython.core.debugger import Tracer
from scipy.io import wavfile


# Spectrogram CNN used for semantic embedding task
# More info see Harwath & Glass 2016 paper
# Written by Liming Wang, Apr. 17, 2017
# Modification:
# 1) Apr. 17: change the output of the scnn function to return all the network parameters,
#   so we do not need to train the network every time before testing

# Use off-the-shelf package for mel frequency spectrogram (not MFCC) for now, may write one myself at some point
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

def loaddata(ntr, ntx):
    mfcc = MFCC()
    dir_info = '../data/flickr_audio/'
    filename_info = 'wav2capt.txt'
    
    dir_sp = '../data/flickr_audio/wavs/'
    dir_penult = '../data/vgg_flickr8k_nnet_penults/'
    
    captions_tr = []
    im_tr = []
    captions_tx = []
    im_tx = []
    Leq = 1024
    with open(dir_info+filename_info, 'r') as f:
        for i in range(ntr):
            # Load the filenames of the files storing the audio captions and its corresponding vgg16 feature
            cur_info = f.readline()
            cur_info_parts = cur_info.rstrip().split()
            sp_name = cur_info_parts[0]
            caption_info = wavfile.read(dir_sp+sp_name)
            caption_time = caption_info[1]
            # Covert the audio into spectrogram
            caption = mfcc.sig2logspec(caption_time)
            # Transpose the caption data
            if caption.shape[0] > caption.shape[1]:
                caption = np.transpose(caption)
            
            # Equalize the length
            if caption.shape[1] < Leq:
                nframes = caption.shape[1]
                nmf = caption.shape[0]
                #print('240:', nframes, (Leq-nframes)/2)
                caption_new = np.zeros((nmf, Leq))
                caption_new[:, round((Leq-nframes)/2):round((Leq-nframes)/2)+nframes] = caption
            else:
                if caption.shape[1] > Leq:
                    nframes = caption.shape[1]
                    nmf = caption.shape[0]
                    caption_new = np.zeros((nmf, Leq))
                    caption_new = caption[:, round((nframes-Leq)/2):round((nframes-Leq)/2)+Leq]
            #print('248:', caption_new)
            captions_tr.append(caption_new)
            # Remove the .jpg# at the end of the file to .npz format, which is used to store vgg16 feature
            im_name_raw = cur_info_parts[1]
            im_name_parts = im_name_raw.split('.')
            im_name = im_name_parts[0]
            # Load the softmax activations of the images, store them into an array
            data = np.load(dir_penult+im_name+'.npz')
            cur_penult = data['arr_0']
            im_tr.append(cur_penult)
            if i%10:
                print('Finish loading', 100*i/ntr, 'percent of training data')

        for j in range(ntx):
            # Load the image names and the image captions, break the captions into words and store in a list
            cur_info = f.readline()
            cur_info_parts = cur_info.rstrip().split()
            sp_name = cur_info_parts[0]
            caption_info = wavfile.read(dir_sp+sp_name)
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
                print('274:', nframes, (Leq-nframes)/2)
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
            data = np.load(dir_penult+im_name+'.npz')
            cur_penult = data['arr_0']
            im_tx.append(cur_penult)
            if j % 10:
                print('Finish loading', 100*j/ntx, 'percent of test data')



    captions_tr = np.array(captions_tr)
    captions_tx = np.array(captions_tx)
    im_tr = np.array(im_tr)
    im_tx = np.array(im_tx)
    np.savez('captions.npz', captions_tr, captions_tx)
    np.savez('images.npz', im_tr, im_tx)
    return captions_tr, captions_tx, im_tr, im_tx


# Load the mfcc data computed by the matlab file loadTIMIT.m
#print('Line 341:', ntr)
#print('Line 342:', ntx)
Fs = 16000;
nlabel = 61;
#nphn = 61;
caption_name = 'captions.npz'
image_name = 'images.npz'
ntr = 10000
ntx = 100

if len(sys.argv) >= 3:
    ntr = int(sys.argv[1])
    ntx = int(sys.argv[2])
    if len(sys.argv) >= 5:
        caption_name = sys.argv[3]
        image_name = sys.argv[4]


if os.path.isfile(caption_name):
    data_caption = np.load(caption_name)
    captions_tr = data_caption['arr_0'][0:ntr]
    captions_tx = data_caption['arr_1'][0:ntx]
    data_im = np.load(image_name)
    im_tr = data_im['arr_0'][0:ntr]
    im_tx = data_im['arr_1'][0:ntx]
    print('Line 333: ', captions_tr.shape)

else:
    captions_tr, captions_tx, im_tr, im_tx = loaddata(ntr, ntx)
print('Line 335: finish loading data')
nframes = captions_tr[0].shape[1]# number of frames in each frequency channel
nmf = captions_tr[0].shape[0]
#slicelen = 16
#factor = round(wavlen/slicelen)
npenult_vgg = 4096
nembed = 1024
#nembed = 256
# Overwrite nframes for test purpose
#nframes = nembed


print('Line 346 dimension of the training data:', nmf, nframes)


# Load flikr8k audio data
X_tr = captions_tr
X_tx = captions_tx
#X_tr = np.random.uniform(low=0, high=1, size=(ntr, nmf, nframes))

# Load penultimate layer data of the vgg net
Z_tr_vgg = im_tr
Z_tx_vgg = im_tx
#Z_tr_vgg = np.random.uniform(low=0, high=1, size=(ntr, npenult_vgg))

#Z_tx[n][z_str_tx[n]-1] = 1
#print('Loaded', n/ntx*100, '% of test data...')
# Length of context, meaning a context window that encloses +-ncontext frames near the current frame
#ncontext = 9;
#nwin = 2*ncontext+1;
#nwin = 20;

learn_rate = 1e-5;
# Stack the mfccs to form a context window. transpose each feature in the X_batch to meet the input requirement of
# contextwin
#nreduce = 0;
#X_stack_tr, Z_stack_tr = contextwin(np.transpose(X_tr, [0, 2, 1]), Z_tr, ncontext, nreduce)
#print('line 31:', X_stack_tr.shape)
#X_stack_tx, Z_stack_tx = contextwin(np.transpose(X_tx, [0, 2, 1]), Z_tx, ncontext, nreduce)

#print(X_stack_tr.shape)
def weight_variable(dims):
    w = tf.Variable(tf.random_normal(dims, stddev=0.01))
    return w

def bias_variable(dims):
    b = tf.Variable(tf.random_normal(dims, stddev=0.01))
    return b

def scnn(X_in, J, N, D):
    # Assume 4 layers
    # Filter dimension: 1 x filter length x num input channels x num output channels
    # X_in, Y: input data and labels respectively
    # J: channel dimension of the architecture of TDNN
    # N: filter order in each channel
    # D: dimension of input node at each layer
    # More see Harwath et al. 2015 & 2016
    # Created by Liming Wang on April. 4th, 2017
    
    # Mean subtraction with mean spectrogram estimated over the entire traininig set
    X_mean = tf.reduce_mean(X_in)
    X_zm = X_in - X_mean
    X_zm = X_in
    w_in = weight_variable([1, N[0]+1, J[0], J[1]])
    b_in = bias_variable([J[1]])
    
    w_hidden1 = weight_variable([1, N[1]+1, J[1], J[2]])
    b_hidden1 = bias_variable([J[2]])
    
    w_hidden2 = weight_variable([1, N[2]+1, J[2], J[3]])
    b_hidden2 = bias_variable([J[3]])
    
    # Initialize softmax layer
    w_out = weight_variable([J[3], J[4]])
    b_out = bias_variable([J[4]])
    
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
    
    #Z_layer2 = tf.nn.conv2d(X_layer1, W_hidden, strides=[1,1,1,1], padding='VALID') + b_hidden
    #X_layer2 = tf.nn.sigmoid(Z_layer2)
    #X_layer2 = tf.nn.relu(Z_layer2)
    #print('37:', tf.shape(X_layer2_re))
    #Z_out = tf.nn.conv2d(X_layer2, W_out, strides=[1,1,1,1], padding='VALID') + b_out
    #a_out = tf.matmul(h4_re, w_out) + b_out
    a_out = tf.matmul(h4_ren, w_out) + b_out
    Y_pred = tf.nn.softmax(a_out)
    
    pmtrs = [w_in, b_in, w_hidden1, b_hidden1, w_hidden2, b_hidden2, w_out, b_out]
    #D1 = tf.nn.sigmoid(Z_out)
    return Y_pred, h4_ren, h3, pmtrs
#return Y_pred, h4_re, h3, w_in

# Build the computation graph
J = [nmf, 64, 512, 1024, nlabel]
#J = [nmf, 64, 128, 256, nlabel]
# Order of filters
N = [4, 24, 24]
D = [nframes, nframes/2, nframes/4, nframes/4]
#D = [nwin, nwin-N[0], nwin-N[0]-N[1]];#[16, 12, 10];
w_embed = weight_variable([npenult_vgg, nembed])
b_embed = bias_variable([nembed])
X_in = tf.placeholder(tf.float32, shape=[None, 1, nframes, nmf])

# Map the penultimate layer of the VGG to embedding space
Z_penult_vgg = tf.placeholder(tf.float32, shape=[None, npenult_vgg])
Z_embed_vgg = tf.matmul(Z_penult_vgg, w_embed) + b_embed

Z_pred, Z_embed_sp, Z_penult_sp, pmtrs = scnn(X_in, J, N, D)
'''in_w = tdnn_pmtrs[0]
    in_b = tdnn_pmtrs[1]
    h1_w = tdnn_pmtrs[2]
    h1_b = tdnn_pmtrs[3]
    o_w = tdnn_pmtrs[4]
    o_b = tdnn_pmtrs[5]'''
#s_i = tf.max(tf.zeros(nbatch, 1), tf.subtract(s_score, tf.add(tf.diag(s_score), tf.eye(nbatch))))
#s_c = tf.max(tf.zeros, tf.subtract(s_score, tf.add(tf.diag(s_score), tf.eye(nbatch))))

# Compute the similarity scores
s_a = tf.matmul(Z_embed_sp, tf.transpose(Z_embed_vgg))
s = tf.nn.relu(s_a)
s_p = tf.diag_part(s)
# Maximum margin cost function
cost = tf.reduce_sum(tf.nn.relu(s-s_p+1)+tf.nn.relu(tf.transpose(s)-s_p+1))#*(1-tf.cast(tf.equal(s, s_p), tf.float32))))
#cost = tf.reduce_sum(tf.reduce_max(tf.nn.relu(s-s_p+1), reduction_indices=1)+tf.reduce_max(tf.nn.relu(tf.transpose(s)-s_p+1), reduction_indices=1))
ds = s-s_p
#cost = tf.reduce_sum(s-s_p)
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cost)
#right = tf.equal(tf.argmax(Z_pred, 1), tf.argmax(Z_in, 1))
#accuracy = tf.reduce_mean(tf.cast(right, tf.float32));

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train the ANN model (TDNN) using minibatch gradient descent

batch_size = 128;
nbatch = int(ntr/batch_size);
niter = 20;

tr_accuracy = np.zeros([niter,])
dev_accuracy = np.zeros([niter, ])
for t in range(niter):
    #randidx = np.array([0, 1, 2])
    randidx = np.argsort(np.random.normal(size=(ntr,)), 0)
    X_tr_rand = X_tr[randidx]
    Z_tr_vgg_rand = Z_tr_vgg[randidx]
    for i in range(nbatch):
        #randidices = np.argsort(np.random.normal(size=[batch_size]))
        #X_batch = X_stack_tr[randidx[i]*batch_size:(randidx[i]+1)*batch_size]
        #Z_batch = Z_stack_tr[randidx[i]*batch_size:(randidx[i]+1)*batch_size]
        #X_batch = X_tr_rand[randidx[i]*batch_size:(randidx[i]+1)*batch_size]
        #print('Line 246 dimension of the training data:', X_batch.shape[0])
        X_batch = X_tr_rand[i*batch_size:(i+1)*batch_size]
        #X_batch = X_tr[i*batch_size:(i+1)*batch_size]
        Z_batch = Z_tr_vgg_rand[i*batch_size:(i+1)*batch_size]
        #Z_batch = Z_tr[i*batch_size:(i+1)*batch_size]
        # Recall the input for conv2d is of shape batch x input height x input width x # of channels
        print(X_batch.shape)
        X_batch_4d = np.reshape(X_batch, [batch_size, 1, nframes, nmf])
        sess.run(train_step, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
        if i % 10 == 0:
            #Z_p = sess.run(Z_pred, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            #X_tr_4d = X_tr.reshape([ntr, 1, nframes, nmf])
            # Evaluate the model with the top 10 image matching error rate
            #similarity = sess.run(s, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            similarity = sess.run(s, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            #similarity_p = sess.run(s_p, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #cur_cost = sess.run(cost, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            cur_cost = sess.run(cost, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            # Find the indices of the images with the top 10 similarity score
            ntop = 10
            top_indices = np.zeros((ntop, batch_size))
            for k in range(ntop):
                cur_top_idx = np.argmax(similarity, axis=1)
                top_indices[k] = cur_top_idx
                # To leave out the top values that have been determined and the find the top values for the rest of the indices
                for l in range(batch_size):
                    similarity[l][cur_top_idx[l]] = -1;
            #similarity[cur_top_idx] = -1;
            # Find if the image with the matching index has the highest similarity score
            dev = abs(top_indices - np.linspace(0, batch_size-1, batch_size))
            min_dev = np.amin(dev, axis=0)
            print('current deviation from correct indices for training:', min_dev)
            # Count the number of correct matching by counting the number of 0s in dev
            tr_accuracy[t] = np.mean((min_dev==0))
            #tr_accuracy[t] = sess.run(accuracy, feed_dict={X_in:X_tr_4d, Z_in:Z_tr})
            #loss = sess.run(cross_entropy, feed_dict={X_in:X_tr_4d, Z_in:Z_tr})
            #cur_z_sp = sess.run(Z_embed_sp, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #cur_z_im = sess.run(Z_embed_vgg, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #cur_sp = sess.run(s_p, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #cur_ds = sess.run(ds, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            cur_z_sp = sess.run(Z_embed_sp, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            cur_z_im = sess.run(Z_embed_vgg, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            cur_s = sess.run(s, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            cur_ds = sess.run(ds, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            print('Training similarity score is:\n', cur_s)
            print('Iteration', t, 'at batch', i)
            print('Training accuracy is: ', tr_accuracy[t])

            X_dev_4d = np.reshape(X_tx[0:200], [200, 1, nframes, nmf])
            Z_dev = Z_tx_vgg[0:200]
            similarity = sess.run(s, feed_dict={X_in:X_dev_4d, Z_penult_vgg:Z_dev})
            #similarity_p = sess.run(s_p, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #cur_cost = sess.run(cost, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            cur_cost = sess.run(cost, feed_dict={X_in:X_dev_4d, Z_penult_vgg:Z_dev})
            # Find the indices of the images with the top 10 similarity score
            ntop = 10
            top_indices = np.zeros((ntop, 200))
            for k in range(ntop):
                cur_top_idx = np.argmax(similarity, axis=1)
                top_indices[k] = cur_top_idx
                # To leave out the top values that have been determined and the find the top values for the rest of the indices
                for l in range(batch_size):
                    similarity[l][cur_top_idx[l]] = -1;
            #similarity[cur_top_idx] = -1;
            # Find if the image with the matching index has the highest similarity score
            dev = abs(top_indices - np.linspace(0, 199, 200))
            min_dev = np.amin(dev, axis=0)
            #print('current deviation from correct indices for dev test:', min_dev)
            dev_accuracy[t] = np.mean((min_dev==0))
            print('Development accuracy is: ', dev_accuracy[t])

            #scnn_pmtrs = sess.run(pmtrs, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            #w_embed = sess.run(w_embed, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #b_embed = sess.run(b_embed, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
            #print('similarity score:', np.dot(cur_z_sp, np.transpose(cur_z_im)))
            
            #print('Current audio caption mel-freq feature:\n', X_batch[:, :, 200:600])
            #print('Current audio caption penultimate feature:\n', cur_z_sp)
            #print('Current image feature:\n', Z_tr_vgg)
            
            #print('Training similarity score is:\n', similarity)
            #print('Training similarity score of the correct pair at iteration', t, 'batch', i*10, 'is:\n', similarity_p)
            #print('Training cost is:', cur_cost)
            #print('current deviation from correct indices:', min_dev)
            #print('current correct pair score:\n', cur_sp)
            #print('current similarity margin:\n', cur_ds)
            #print('Top 10 indices is:\n', top10_indices)
            print('\n')

# Save training accuracy
np.savetxt('train_accuracy_scnn.txt', tr_accuracy)

# Test the ANN model (SCNN)
X_tx_4d = X_tx.reshape([ntx, 1, nframes, nmf])
similarity = sess.run(s, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx_vgg})
#X_tx_4d = X_stack_tx.reshape([ntx*(nframes-2*nreduce), 1, nwin, nmf])
#test_accuracy = sess.run(accuracy, feed_dict={X_in:X_tx_4d, Z_in:Z_tx})
ntop = 10
top_indices = np.zeros((ntop, ntx))
for k in range(ntop):
    # Find the most similar image feature of the speech feature on the penultimate feature space
    cur_top_idx = np.argmax(similarity, axis=1)
    top_indices[k] = cur_top_idx
    # To leave out the top values that have been determined and the find the top values for the rest of the indices
    for l in range(ntx):
        similarity[l][cur_top_idx[l]] = -1
        #similarity[cur_top_idx] = -1;

# Find if the image with the matching index has the highest similarity score
#dev = abs(np.transpose(np.transpose(top10_indices) - np.linspace(0, ntr-1, ntr)))
dev = abs(top_indices - np.linspace(0, ntx-1, ntx))
min_dev = np.amin(dev, axis=0)
# Count the number of correct matching by counting the number of 0s in dev
test_accuracy = np.mean(min_dev == 0)
print('Test accuracy is: ', str(test_accuracy))

#runtime = timeit() - begin_time
#print('Total runtime:', runtime)
# Save the parameters of the scnn
X_tr_4d = X_tr.reshape([ntr, 1, nframes, nmf])
scnn_pmtrs = sess.run(pmtrs, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
w_embed = sess.run(w_embed, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
b_embed = sess.run(b_embed, feed_dict={X_in:X_tr_4d, Z_penult_vgg:Z_tr_vgg})
np.savez('scnn_pmtrs.npz', scnn_pmtrs)
np.savez('vgg_pmtrs.npz', [w_embed, b_embed])
'''in_w_opt = sess.run(in_w, feed_dict={X_in:X_tr, Z_in:Z_tr})
    in_b_opt = sess.run(in_b, feed_dict={X_in:X_tr, Z_in:Z_tr})
    h1_w_opt = sess.run(h1_w, feed_dict={X_in:X_tr, Z_in:Z_tr})
    h1_b_opt = sess.run(h1_b, feed_dict={X_in:X_tr, Z_in:Z_tr})
    o_w_opt = sess.run(o_w, feed_dict={X_in:X_tr, Z_in:Z_tr})
    o_b_opt = sess.run(o_b, feed_dict={X_in:X_tr, Z_in:Z_tr})
    pmtrs = [in_w_opt, in_b_opt, h1_w_opt, h1_b_opt, o_w_opt, o_b_opt]
    pmtrs_name = ['in_w_opt', 'in_b_opt', 'h1_w_opt', 'h1_b_opt', 'o_w_opt', 'o_b_opt']
    for i in range(npmtr):
    filename = pmtrs_name[i]+'.txt';
    with open(filename, 'wb') as f:
    cur_pmtr = pmtrs[i]
    shp = shape(cur_pmtr)
    if len(shp) <= 2:
    np.savetxt(f, cur_pmtr)
    else:
    if len(shp) == 3:
    for j in range(shp[3]):
    np.savetxt(f, cur_pmtr[:, :, j])
    else:
    if len(shp) == 4:
    for j in range(shp[3]):
    for k in range(shp[4]):
    np.savetxt(f, cur_pmtr[:, :, j, k])'''


