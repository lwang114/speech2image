# -*- coding: utf-8 -*-
import numpy as np
import h5py
import PIL
import scipy.io.wavfile
import scipy.misc
import cv2, numpy as np
from mfcc import *
from vgg16 import *
  
def set_nframes_to_fixed_size(logspec, target_framecount):
    '''If logspec has more than target_framecount frames, truncate it.
    If it has less, zero-pad at beginning and end.  Return the result.'''
    nframes = logspec.shape[1]
    margin = int((nframes-target_framecount)/2)
    logspec_new = np.zeros((logspec.shape[0], target_framecount))
    if margin < 0:
        logspec_new[:, (-margin):(nframes-margin) ] = logspec
    if margin > 0:
        logspec_new = logspec[:, margin:(target_framecount+margin)]
    return logspec_new


if __name__ == '__main__':
  im_path = '/home/lwang114/data/flickr/Flicker8k_Dataset/'
  aud_path = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_audio/wavs/'
  # Instantiate vgg model and mfcc class
  mfcc = MFCC()
  sess = tf.Session()
  imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
  vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

  image_capt_file = 'flickr_im_capt_pairs.txt'
  with open(image_capt_file, 'r') as f:
    image_capt_pairs = f.read().strip().split('\n')
      
  # Create a dictionary to map the id of a file to its purpose in the karpathy split
  split_path_tr = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_train.npz'
  split_path_val = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_val.npz'
  split_path_tx = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_test.npz'
  npz_tr = np.load(split_path_tr)
  tr_files = npz_tr.keys() 
  npz_tx = np.load(split_path_tx)
  tx_files = npz_tx.keys()
  npz_val = np.load(split_path_val)
  val_files = npz_val.keys()
  idToPurpose = dict()
  for f in tr_files:
    idToPurpose[f.split('/')[-1].split('_')[0]] = 'train'
  for f in tx_files:
    idToPurpose[f.split('/')[-1].split('_')[0]] = 'test'
  for f in val_files:
    idToPurpose[f.split('/')[-1].split('_')[0]] = 'val'

  # Generate fbank and penult features and store them in an h5 file; use a karpathy split
  nf = len(image_capt_pairs)
  fbank_tr = []
  vgg_tr = []
  wrd_tr = []
  fbank_tx = []
  vgg_tx = []
  wrd_tx = []
  fbank_val = []
  vgg_val = []
  wrd_val = []
  for pair in image_capt_pairs:
    info = pair.split(' ')
    aud_file = aud_path + info[1].split('/')[-1].split('.')[0] + '.wav'
    print(aud_file)
    im_file = im_path + info[0].split('/')[-1]
    bb = info[3:7]
    timing = info[7:]
    wrd = info[2] 
    # Create a group for the current word
    cur_id = aud_file.split('/')[-1].split('_')[0]
    
    # Extract fbank feature for the audio
    start = float(timing[0])
    end = float(timing[1])
    Fs, aud = scipy.io.wavfile.read(aud_file)
    fbank = set_nframes_to_fixed_size(mfcc.sig2logspec(aud[int(start*Fs):int(end*Fs)]).T, 100)
    print(fbank.shape)    
    # Extract vgg penult feature for the visual objects
    bb_int = [int(pos) for pos in bb]
    im_obj = PIL.Image.open(im_file)
    im = np.array(im_obj)
    reg = im[bb_int[1]:bb_int[3], bb_int[0]:bb_int[2], :]
    reg_fixsize = scipy.misc.imresize(reg, (224, 224, 3))
    penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
    
    # Store the features to the current group
    if idToPurpose[cur_id] == 'train':
      fbank_tr.append(fbank)
      vgg_tr.append(penult)
      wrd_tr.append(wrd)    
    elif idToPurpose[cur_id] == 'test':
      fbank_tx.append(fbank)
      vgg_tx.append(penult)
      wrd_tx.append(wrd)
    else:
      fbank_val.append(fbank)
      vgg_val.append(penult)
      wrd_val.append(wrd)

  # Create a HDF-5 file  
  h5_feat = h5py.File('flickr_wrd_fbank_penult.h5', 'w')

  tr_grp = h5_feat.create_group('train')
  tx_grp = h5_feat.create_group('test')
  val_grp = h5_feat.create_group('val')

  tr_grp.create_dataset('fbank', data=np.array(fbank_tr))
  tr_grp.create_dataset('vgg_penult', data=np.array(vgg_tr))
  #tr_grp.create_dataset('wrd', data=wrd_tr)
  tx_grp.create_dataset('fbank', data=np.array(fbank_tx))
  tx_grp.create_dataset('vgg_penult', data=np.array(vgg_tx))
  #tx_grp.create_dataset('wrd', data=wrd_tx)
  val_grp.create_dataset('fbank', data=np.array(fbank_val))
  val_grp.create_dataset('vgg_penult', data=np.array(vgg_val))
  #val_grp.create_dataset('wrd', data=wrd_val)

  h5_feat.close()
