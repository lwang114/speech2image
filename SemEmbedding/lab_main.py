########################################################################
# Jelinek Speech and Language Technology workshop 2017
# speech2image lab
# Mark Hasegawa-Johnson
# First draft, 2017 June 10
#
# Things you need to download:
# 1. tensorflor, numpy, and scipy if you don't have them.
# 2. The files vgg16.py, imagenet_classes.py, and vgg16_weights.npz from
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
# 3. The file mfcc.py from
# http://www.cs.cmu.edu/~dhuggins/Projects/pyphone/sphinx/mfcc.py
########################################################################

# Tensorflow, numpy, scipy, os
import tensorflow as tf
import numpy as np
# Davi Frossard's TensorFlow implementation of VGG16 ImageNet object detector
from vgg16 import vgg16
# Liming Wang's implementation of Speech CNN
from lscnn import LSCNN

if __name__ == '__main__':

    ###########################################################################
    # How many images shall we use?
    n_images = 10
    n_audios = 5*n_images
    
    ###########################################################################
    # First, read the input images, compute VGG feature files, and write them
    imagefile_path = '../data/flickr8k/Flicker8k_Dataset'
    imagefile_list = os.listdir(imagefile_path)
    vfeatfile_list = [ x[:-4] for x in imagefile_list ]

    # Create a new default graph, within which the VGG16 network will go
    with tf.Graph().as_default() as g_vgg16:
        vgg16_session = tf.Session()
        vgg16_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg16_nnet = vgg16(vgg16_input, '../../vgg16/vgg16_weights.npz', vgg16_session)
    
        for n in range(0,n_images):
            img1 = imread(imagefile_path+'/'+imagefile_list[n], mode='RGB')
            img1 = imresize(img1, (224,224))
            fc2 = vgg16_session.run(vgg.fc2, feed_dict={vgg.imgs: [img1]})[0]
            np.savez(vfeatfile_list[n],penults)
        
    ###########################################################################
    # Second, train the LSCNN, and test on the training images
    # Create a separate tensorflow graph to store this network
    with tf.Graph().as_default() as g_lscnn:
        lscnn = LSCNN('train')
        lscnn.load_data(n_audios)
        training_toks = range(n_audios)
        lscnn.train( training_toks )  
        lscnn.save_weights('lscnn_weights') 

        # test on the training data
        testing_toks = training_toks
        lscnn.test( testing_toks )
