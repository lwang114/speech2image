#######################################################################
# Spectrogram CNN used for semantic embedding task
# More info see Harwath & Glass 2016 paper
# Written by Liming Wang, Apr. 17, 2017
# Modification:
# 1) Apr. 17: change the output of the scnn function to return all the
#    network parameters, so we do not need to train the network every
#    time before testing, Liming Wang
# 2) June 10: Separate train/test code and network architecture, so it can be
#    more easily used in a tutorial, Mark Hasegawa-Johson
#######################################################################
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import os
import sys
from timeit import timeit
from scipy.io import wavfile
# You need to download mfcc.py from David Huggins-Daines webpage
from mfcc import MFCC

##################################################################
# CONVENIENCE FUNCTIONS
def weight_variable(dims):
    '''A convenience function that creates a weight matrix,
    this is used by lscnn.scnn() and lscnn.vggembed()'''
    w = tf.Variable(tf.random_normal(dims, stddev=0.01))
    return w

def bias_variable(dims):
    '''A convenience function that creates a bias vector,
    this is used by lscnn.scnn() and lscnn.vggembed()'''
    b = tf.Variable(tf.random_normal(dims, stddev=0.01))
    return b

def load_logspec_from_wavfile(filename, mfcc):
    '''Load a waveform, convert it into logmelspectral features,
    transpose it so it is numfreqs x numframes, and return it.
    Usage:
    from mfcc import MFCC
    mfcc = MFCC()
    logspec = load_logspec_from_wavefile(filename, mfcc)'''
    wave_info = wavfile.read(filename)
    logspec = mfcc.sig2logspec(wave_info[1])
    logspec = np.transpose(logspec)  # Transpose to make it nmf x nframes
    return logspec

def set_nframes_to_fixed_size(logspec, target_framecount):
    '''If logspec has more than target_framecount frames, truncate it.
    If it has less, zero-pad at beginning and end.  Return the result.'''
    nframes = logspec.shape[1]
    margin = int((nframes-target_framecount)/2)
    logspec_new = np.zeros((logspec.shape[0], target_framecount))
    if margin < 0:
        logspec_new[:, (-margin):(nframes-margin) ] = logspec
    if margin > 0:
        logspec_new = caption[:, margin:(target_framecount+margin)]
    return logspec_new

##################################################################
# The Latent Speech CNN Class
class LSCNN:
    '''Latent-from-Speech Convolutional Neural Network
    Written by Liming Wang, refactored by Mark Hasegawa-Johnson,
    based on a paper by David Harwath and James Glass.

    USAGE:
    lscnn = LSCNN(nchan_scnn, width_scnn, strid_scnn, use_case)
    lscnn.load_data(num_toks, wav2capt_filename, speech_dir, vgg16_dir)
    ... then either ...
    lscnn.train( training_toks )
    lscnn.save_weights(weights_file) 
    ... or ...
    lscnn.load_weights(weights_file)
    lscnn.test( testing_toks )

    See the description of each function for information about parameters.
'''
    def __init__(self, use_case, nchan_scnn=[40,64,512,1024], width_scnn=[5,25,25], strid_scnn=[2,2,256], nnode_imag=4096):
        '''lscnn = LSCNN(nchan_scnn, width_scnn, strid_scnn, use_case)

        PARAMETERS:
        use_case [string] = either the word 'train' or 'test'.
          This string determines what type of cost function is added to the 
          graph.
        nchan_scnn [list]: number of channels in each CNN layer.
          Default = [40, 64, 512, 1024]
          The 0th element specifies the number of mel-frequency bins.
          The last element also specifies the dimension of the semantic
          embedding space. Dimension of the VGG16 input features
          is assumed to be 8192.
        width_scnn [list]: temporal span of each CNN filter.
          Default: [5, 25, 25]
        strid_scnn [list]: stride to use in max pooling at each layer.
          Default: [2, 2, 256]
          An input spectrogram is forced to have a length equal to 
          exactly the product of these.  So if you want to change the
          standard length of the input spectrogram, the way to do it is to
          change strid_scnn[-1].
        nnode_vgg [int]: size of the image feature input vector.
          Default: 4096
        '''
        self.nembed = nchan_scnn[-1] # semantic embedding dimension
        self.nfilts = nchan_scnn[0]  # number of mel-frequency filters
        self.target_framecount = np.prod(np.array(strid_scnn)) # Leq
        self.nnode_imag = nnode_imag
        print('SCNN: {},{} speech, {} image to {} semantic'.format(self.nfilts,self.target_framecount,nnode_imag,self.nembed))
        
        # Build the computation graph for the audio CNN
        self.pmtrs = self.scnn(nchan_scnn, width_scnn, strid_scnn)

        # Build the graph for the linear transformation VGG features
        self.pmtrs += self.vggembed(nnode_imag, self.nembed)

        # Define the similarity metric, used for both training and testing
        self.similarity()
        
        # Training setup
        if use_case == 'train':
            self.create_train_step() # define the training step

        # Create the session, then randomly initialize all variables.
        # If you're going to test a network, you probably want to
        # over-write the random initialization using load_weights
        self.sess = tf.Session() # define the session
        self.init = tf.global_variables_initializer() 
        self.sess.run(self.init) # generate initial values

    ##############################################################
    # This loads all data into RAM, all at once
    # Each spectrogram is 40x1024x(double) = 320k
    # Loading all of flickr8k (40000 files) will consume 12.8G
    # MSCOCO probably can't be loaded in this way
    def load_data(self, numtoks, wav2capt_filename='../../../data/flickr8k/flickr_audio/wav2capt.txt', dir_sp='../../../data/flickr8k/flickr_audio/wavs/', dir_penult='../../vgg16/vgg_flickr8k_nnet_penults/'):
        '''lscnn.load_data(num_toks, wav2capt_filename, speech_dir, vgg16_dir)
        numtoks [integer] = number of tokens to load
        wav2capt_filename [string] = filename containing wave to image map
        speech_dir [string] = directory containing speech files
        dir_penult [string] = dir containing VGG16 features for images'''
        mfcc = MFCC(self.nfilts)
        afeats = []
        vfeats = []
        with open(wav2capt_filename, 'r') as f:
            for i in range(numtoks):
                cur_info = f.readline().rstrip().split()
                # Load the filenames of the files storing the audio captions
                caption = load_logspec_from_wavfile(dir_sp+cur_info[0], mfcc)
                caption_new = set_nframes_to_fixed_size(caption, self.target_framecount)
                afeats.append(caption_new)
                # Remove the .jpg# at the end of the file to .npz format
                im_name = os.path.splitext(cur_info[1])
                # Load the VGG penults, store them into an array
                data = np.load(dir_penult+im_name[0]+'.npz')
                cur_penult = data['arr_0']
                vfeats.append(cur_penult)

            self.afeats = np.array(afeats)
            self.vfeats = np.array(vfeats)
            np.savez('captions.npz', self.afeats)
            np.savez('images.npz', self.vfeats)
        
    #################################################
    def save_weights(self, weight_file):
        '''lscnn.save_weights(weight_file)
        weight_file [string]: save weights in weight_file + ".npz"'''
        saveable = [ self.sess.run(v) for v in self.pmtrs ]
        np.savez(weight_file, saveable)

    #################################################
    def load_weights(self, weight_file):
        '''lscnn.load_weights(weight_file)
        weight_file should be an npz file containing self.pmtrs'''
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            self.sess.run(self.pmtrs[i].assign(weights[k]))

    #########################################################
    def scnn(self, nchan_scnn, width_scnn, strid_scnn):
        '''lscnn.scnn(nchan_scnn, width_scnn, strid_scnn).
        Create tf.Variables and tf.Operations that implement a
        speech CNN, mapping spectrograms to the semantic embedding.
        nchan_scnn [list] = specifies number of channels in each layer
        width_scnn [list] = specifies width of the filter in each layer
        strid_scnn [list] = specifies stride of max pooling in each layer.
        Created by Liming Wang on April. 4th, 2017,
        based on Harwath et al. 2015 & 2016'''

        # Create a placeholder specifying the input
        self.X_in = tf.placeholder(tf.float32, shape=[None,1,self.target_framecount,self.nfilts])
        
        # Subtract mean spectrogram estimated over the entire training set
        self.X_mean = tf.reduce_mean(self.X_in)
        self.X_zm = self.X_in - self.X_mean

        parameters = []

        nlayers = len(width_scnn)
        
        # Format each of the convolutional layers
        # w[layer] = 1 x width_scnn x nchan_scnn[layer] x nchan_scnn[layer+1]
        # a_node[layer] = ntoks x 1 x nframes[layer] x nchan_scnn[layer]
        self.b = []
        self.a_conv = []
        self.a_pool = []
        self.a_node = [ self.X_in ]
        self.b = []
        self.w = []
        for layer in range(nlayers):
            self.w.append(weight_variable([1, width_scnn[layer], nchan_scnn[layer], nchan_scnn[layer+1]]))
            self.b.append(bias_variable([nchan_scnn[layer+1]]))
            self.a_conv.append(tf.nn.conv2d(self.a_node[layer], self.w[layer], strides=[1,1,1,1], padding='SAME') + self.b[layer])
            if layer < nlayers-1:
                self.a_pool.append(tf.nn.max_pool(self.a_conv[layer],ksize=[1,1,2*strid_scnn[layer],1],strides=[1,1,strid_scnn[layer],1],padding='SAME'))
            else:
                self.a_pool.append(tf.nn.max_pool(self.a_conv[layer],ksize=[1, 1,strid_scnn[layer],1], strides=[1,1,1,1], padding='VALID'))

            self.a_node.append(tf.nn.relu(self.a_pool[layer]))
            parameters += [ self.w[layer], self.b[layer] ]
    
        ############################################################
        # Reshape the last layer into a vector, and L2 normalize
        self.a_vec = tf.reshape(self.a_node[-1], [-1, nchan_scnn[-1]])
        self.a_embed = tf.nn.l2_normalize(self.a_vec, dim=1)    
        
        return parameters

    ##########################################################
    def vggembed(self, npenult_vgg, nembed):
        '''lscnn.vggembed(npenult_vgg, nembed)
        Create tf.Variables and tf.Operations to linearly transform
        VGG16 feature vector into a semantic embedding space.
        npenult_vgg = dimension of the input VGG16 feature vector
        nembed = dimension of the semantic embedding.'''
        self.v_in = tf.placeholder(tf.float32, shape=[None, npenult_vgg])
        self.w_embed = weight_variable([npenult_vgg, nembed])
        self.b_embed = bias_variable([nembed])
        self.v_embed = tf.matmul(self.v_in, self.w_embed) + self.b_embed
        return [ self.w_embed, self.b_embed ]

    ###########################################################
    def similarity(self):
        '''Compute dot product between audio and video embeddings'''
        self.s_a = tf.matmul(self.a_embed, tf.transpose(self.v_embed))
        self.s = tf.nn.relu(self.s_a)
        self.s_p = tf.diag_part(self.s)
        self.cost = tf.reduce_sum(tf.nn.relu(self.s-self.s_p+1)+tf.nn.relu(tf.transpose(self.s)-self.s_p+1))

    ############################################################
    def create_train_step(self, learn_rate = 1e-5):
        self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.cost)

    #############################################################
    def train(self, list_training_toks):
        '''lscnn.train( training_toks )
        training_toks [iterable] = indices of tokens to use for training'''

        training_toks = np.array(list_training_toks) # so it can be sorted
        ntr = len(training_toks)
        batch_size = min(ntr,128);
        nbatch = int(ntr/batch_size);
        niter = 20;
        tr_accuracy = np.zeros([niter,])
        dev_accuracy = np.zeros([niter, ])
        for t in range(niter):
            randidx = np.argsort(np.random.normal(size=(ntr,)), 0)
            randsort_toks = training_toks[randidx]
            for i in range(nbatch):
                batch_toks = randsort_toks[i*batch_size:(i+1)*batch_size]
                X_batch = self.afeats[batch_toks]
                Z_batch = self.vfeats[batch_toks]
                print(X_batch.shape)
                X_batch_4d = np.reshape(X_batch, [batch_size, 1, self.target_framecount, self.nfilts])
                self.sess.run(self.train_step, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                if i % 10 == 0:
                    similarity = self.sess.run(self.s, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                    cur_cost = self.sess.run(self.cost, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                    # Find the indices of the images with the top 10 similarity score
                    ntop = 10
                    top_indices = np.zeros((ntop, batch_size))
                    for k in range(ntop):
                        cur_top_idx = np.argmax(similarity, axis=1)
                        top_indices[k] = cur_top_idx
                        # Make sure (k+1)st top is not same as k'th top
                        for l in range(batch_size):
                            similarity[l][cur_top_idx[l]] = -1;
                    dev = abs(top_indices - np.linspace(0, batch_size-1, batch_size))
                    min_dev = np.amin(dev, axis=0)
                    print('current deviation from correct indices for training:', min_dev)
                    # Count the number of correct matching by counting the number of 0s in dev
                    tr_accuracy[t] = np.mean((min_dev==0))
                    cur_z_sp = self.sess.run(self.a_embed, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                    cur_z_im = self.sess.run(self.v_embed, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                    cur_s = self.sess.run(self.s, feed_dict={self.X_in:X_batch_4d, self.v_in:Z_batch})
                    print('Training similarity score is:\n', cur_s)
                    print('Iteration', t, 'at batch', i)
                    print('Training accuracy is: ', tr_accuracy[t])

    ##########################################################
    def test(self, testing_toks):
        '''lscnn.test( testing_toks )
        testing_toks [iterable] = indices of tokens to use for training'''
        nmf = self.nfilts
        nframes = self.target_framecount
        ntx = len(testing_toks)
        X_dev_4d = np.reshape(self.afeats[testing_toks], [ntx, 1, nframes, nmf])
        Z_dev = self.vfeats[testing_toks]
        similarity = self.sess.run(self.s, feed_dict={self.X_in:X_dev_4d, self.v_in:Z_dev})
        cur_cost = self.sess.run(self.cost, feed_dict={self.X_in:X_dev_4d, self.v_in:Z_dev})
        # Find the indices of the images with the top 10 similarity score
        ntop = int(ntx/5)
        top_indices = np.zeros((ntop, ntx))
        for k in range(ntop):
            cur_top_idx = np.argmax(similarity, axis=1)
            top_indices[k] = cur_top_idx
            # To leave out the top values that have been determined
            # and the find the top values for the rest of the indices
            for l in range(batch_size):
                similarity[l][cur_top_idx[l]] = -1;
        # Find if image w/ matching index has highest similarity score
        dev = abs(top_indices - np.linspace(0, 199, 200))
        min_dev = np.amin(dev, axis=0)
        dev_accuracy[t] = np.mean((min_dev==0))
        print('Test accuracy is: ', dev_accuracy[t])
