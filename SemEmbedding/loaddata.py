import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#from IPython.core.debugger import Tracer
from scipy.io import wavfile

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
        captions_tr = np.array(captions_tr)
        im_tr = np.array(im_tr)
        np.savez('captions_tr.npz', 'captions_tr')
        np.savez('images_tr.npz', 'im_tr')
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
        captions_tx = np.array(captions_tx)
        im_tx = np.array(im_tx)
        np.savez('captions_test.npz', 'captions_tx')
        np.savez('images_test.npz', 'im_tx')
    np.savez('captions.npz', captions_tr, captions_tx)
    np.savez('images.npz', im_tr, im_tx)
    return captions_tr, captions_tx, im_tr, im_tx

ntr = int(sys.argv[1])
ntx = int(sys.argv[2])
captions_tr, captions_tx, im_tr, im_tx = loaddata(ntr, ntx)

