# -*- coding: utf-8 -*-
import numpy as np
import h5py
import PIL
import glob
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
  DEBUG = True
  im_path = '/home/lwang114/data/flickr/Flicker8k_Dataset/'
  aud_path = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_audio/wavs/'
  text_path = '/home/lwang114/data/flickr/word_segmentation/'
  concept_path = '/home/lwang114/spring2018/sp2im_word/data/concepts.txt' 

  # Create a concept to id dictionary
  with open(concept_path, 'r') as f:
    concepts = f.read().strip().split('\n')
  nc = len(concepts)
  concept2id = {c:i for i, c in enumerate(concepts)}

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
  tr_files = sorted(npz_tr.keys(), key=lambda x: int(x.split('_')[-1])) 
  npz_tx = np.load(split_path_tx)
  tx_files = sorted(npz_tx.keys(), key=lambda x: int(x.split('_')[-1]))
  npz_val = np.load(split_path_val)
  val_files = sorted(npz_val.keys(), key=lambda x: int(x.split('_')[-1]))
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
  lbl_tr = []
  fbank_tx = []
  vgg_tx = []
  lbl_tx = []
  fbank_val = []
  vgg_val = []
  lbl_val = []

  data_files = tr_files + tx_files + val_files #['/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_audio/wavs/2127207912_9298824e66_3_0'] #
  with open('flickr_sp2im_sentences.txt', 'w') as f:
    f.write('\n'.join(data_files))

  data_files_clean = []

  for df in data_files:
    aud_file = aud_path + '_'.join(df.split('/')[-1].split('_')[:-1]) + '.wav'
    print(aud_file)
    im_file = im_path + '_'.join(df.split('/')[-1].split('_')[:-2]) + '.jpg'
    
    text_file = text_path + '_'.join(df.split('/')[-1].split('_')[:-1]) + '.words'
    print(text_file)
    # Create a group for the current word
    cur_id = aud_file.split('/')[-1].split('_')[0]
  
    # Extract fbank feature for the audio
    lbl = np.zeros((nc,))
    
    if glob.glob(text_file):
      with open(text_file, 'r') as f:  
        infos = f.read().strip().split('\n')
      #if DEBUG:
      #  print(len(infos))
      if len(infos) <= 1:
        continue
      for info in infos:
        wrd = info.split()[0]
        if wrd.lower() in concepts:
          lbl[concept2id[wrd.lower()]] = 1
        elif wrd[:-1].lower() in concepts:
          lbl[concept2id[wrd[:-1].lower()]] = 1
        elif wrd[:-2].lower() in concepts:
          lbl[concept2id[wrd[:-2].lower()]] = 1
    else:
      if DEBUG:
        print('File Do Not Exist')
      continue
    try:
      Fs, aud = scipy.io.wavfile.read(aud_file)
    except:
      continue
    fbank = set_nframes_to_fixed_size(mfcc.sig2logspec(aud).T, 1024)
    if DEBUG:
      print(fbank.shape)    
    # Extract vgg penult feature for the visual objects
    im_obj = PIL.Image.open(im_file)
    im = np.array(im_obj)
    reg_fixsize = scipy.misc.imresize(im, (224, 224, 3))
    penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
    
    # Store the features to the current group; ignore those without known concepts
    if np.sum(lbl) == 0:
      continue
    
    if DEBUG:
        print(np.sum(lbl))
    if idToPurpose[cur_id] == 'train':
      fbank_tr.append(fbank)
      vgg_tr.append(penult)
      lbl_tr.append(lbl)    
    elif idToPurpose[cur_id] == 'test':
      fbank_tx.append(fbank)
      vgg_tx.append(penult)
      lbl_tx.append(lbl)
    else:
      fbank_val.append(fbank)
      vgg_val.append(penult)
      lbl_val.append(lbl)
    data_files_clean.append(df) 

  with open('flickr_sp2im_sentences_cleanup.txt', 'w') as f:
    f.write('\n'.join(data_files_clean))


  # Create a HDF-5 file  
  h5_feat = h5py.File('flickr_sent_fbank_penult_order.h5', 'w')

  tr_grp = h5_feat.create_group('train')
  tx_grp = h5_feat.create_group('test')
  val_grp = h5_feat.create_group('val')

  tr_grp.create_dataset('fbank', data=np.array(fbank_tr))
  tr_grp.create_dataset('vgg_penult', data=np.array(vgg_tr))
  tr_grp.create_dataset('lbl', data=lbl_tr)
  tx_grp.create_dataset('fbank', data=np.array(fbank_tx))
  tx_grp.create_dataset('vgg_penult', data=np.array(vgg_tx))
  tx_grp.create_dataset('lbl', data=lbl_tx)
  val_grp.create_dataset('fbank', data=np.array(fbank_val))
  val_grp.create_dataset('vgg_penult', data=np.array(vgg_val))
  val_grp.create_dataset('lbl', data=lbl_val)

  h5_feat.close()
