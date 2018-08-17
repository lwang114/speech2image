# -*- coding: utf-8 -*-

###################### extract_flickr_sentence2.py #########################
# Description: This script extract the fbank and penultimate features for 
# each image-caption pair, with two additional operations:
# 1) Make synonyms to be refered to as the same concept  
# 2) Merge regions that overlaps too much and use the more frequent concept
# as the label
#
# Author: Liming Wang
# Date: 8. 04. 2018 
############################################################################


import numpy as np
import h5py
import PIL
import glob
import json
import scipy.io.wavfile
import scipy.misc
import cv2, numpy as np
from mfcc import *
from vgg16 import *
  
def IoU(reg1, reg2):
    xmin = min(reg1[0], reg2[0])
    xminmax = max(reg1[0], reg2[0])
    ymin = min(reg1[1], reg2[1])
    yminmax = max(reg1[1], reg2[1])    
    xmax = max(reg1[0] + reg1[2], reg2[0] + reg2[2])
    xmaxmin = min(reg1[0] + reg1[2], reg2[0] + reg2[2])
    ymax = max(reg1[1] + reg1[3], reg2[1] + reg2[3])
    ymaxmin = min(reg1[1] + reg1[3], reg2[1] + reg2[3])
    S_u = (ymax - ymin) * (xmax - xmin)                  
    S_i = (ymaxmin - yminmax) * (xmaxmin - xminmax)
    return S_i/S_u

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
  DEBUG = False
  im_path = '/home/lwang114/data/flickr/Flicker8k_Dataset/'
  aud_path = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_audio/wavs/'
  text_path = '/home/lwang114/data/flickr/word_segmentation/'
  concept_path = '/home/lwang114/spring2018/sp2im_word/data/concepts.txt' 
  concept_path_all = '/home/lwang114/spring2018/sp2im_word/data/concepts_all.txt'

  # Create a concept to id dictionary
  with open(concept_path, 'r') as f:
    concepts = f.read().strip().split('\n')
  with open(concept_path_all, 'r') as f:
    concepts_all = f.read().strip().split('\n')
  
  concept2freq = {}
  for i, c_line in enumerate(concepts_all):
    c_line_parts = c_line.split()
    wrd = c_line_parts[0].lower()
    freq = int(c_line_parts[1])
    if wrd in concept2freq.keys():
      concept2freq[wrd] += freq
    else:
      concept2freq[wrd] = freq  
  
  nc = len(concepts)
  concept2id = {}
  for i, c_line in enumerate(concepts):
    for c in c_line.split():
      concept2id[c] = i

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

  with open('id2purpose.json', 'w') as f:
    json.dump(idToPurpose, f)

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

  with open('im2info.json', 'r') as f:
    im2info = json.load(f)
  data_files_clean = []

  for df in data_files:
    aud_file = aud_path + '_'.join(df.split('/')[-1].split('_')[:-1]) + '.wav'
    #print(aud_file)
    im_file = im_path + '_'.join(df.split('/')[-1].split('_')[:-2]) + '.jpg'
    text_file = text_path + '_'.join(df.split('/')[-1].split('_')[:-1]) + '.words'

    # Find the image id for im2info dict
    im_id = '_'.join(df.split('/')[-1].split('_')[:-1])
    try:
      reg_info = im2info[im_id]
    except:
      continue

    # Find the id of the current image-caption for id2purpose dict
    cur_id = aud_file.split('/')[-1].split('_')[0]
  
    # Initialize the label
    n_maxobj = 5
    lbl = np.zeros((n_maxobj, nc+1))
    # Label for no object is nc+1
    for i in range(n_maxobj):
      lbl[i, nc] = 1
 
    fbank = np.zeros((n_maxobj, 40, 100))
    penult = np.zeros((n_maxobj, 4096))
    # Extract vgg penult feature for the visual objects
    im_obj = PIL.Image.open(im_file)
    im = np.array(im_obj)

    try:
      Fs, aud = scipy.io.wavfile.read(aud_file)
    except:
      print('Audio file not found')
      continue
    
    if glob.glob(text_file):
      print(text_file)
      with open(text_file, 'r') as f:  
        caption_infos = f.read().strip().split('\n')
      #if DEBUG:
      #  print(len(infos))
      if len(caption_infos) <= 1:
        print('Empty segment file')
        continue
      count = 0
      for info in caption_infos:
        wrd = info.split()[0]
        start, end = info.split()[1:]
        start = float(start)
        end = float(end)
        cur_lbl = np.zeros((nc+1,))
        if count < n_maxobj:
          if wrd.lower() in concepts:  
            cur_lbl[concept2id[wrd.lower()]] = 1
            # Search through the whole region info to find the right region
            for reg in reg_info:
              if reg[0] == wrd.lower():
                break
            x, y, w, h = reg[1:5]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            reg_fixsize = scipy.misc.imresize(im[y:y+h, x:x+w, :], (224, 224, 3))
            
            # TODO: Merge regions overlapping too much with the existing regions
            for reg in reg_info:
              if not reg[0] == wrd and reg[0].lower() in concepts:
                x2, y2, w2, h2 = reg[1:5]
                wrd2 = reg[0].lower()
                x2 = int(x2)
                y2 = int(y2)
                w2 = int(w2)
                h2 = int(h2)
                score = IoU([x, y, w, h], [x2, y2, w2, h2])
                if score > 0.5 and concept2freq[wrd2.lower()] > concept2freq[wrd.lower()]:
                  continue

            lbl[count, :] = cur_lbl
            fbank[count, :] = set_nframes_to_fixed_size(mfcc.sig2logspec(aud[int(start*16000):int(end*16000)]).T, 100)
            cur_penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
            penult[count, :] = np.squeeze(cur_penult)
            count += 1
          elif wrd[:-1].lower() in concepts:
            cur_lbl[concept2id[wrd[:-1].lower()]] = 1
            lbl[count, :] = cur_lbl
            fbank[count, :] = set_nframes_to_fixed_size(mfcc.sig2logspec(aud[int(start*16000):int(end*16000)]).T, 100)
            cur_penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
            penult[count, :] = np.squeeze(cur_penult) 
            count += 1
          elif wrd[:-2].lower() in concepts:
            cur_lbl[concept2id[wrd[:-2].lower()]] = 1
            lbl[count, :] = cur_lbl
            fbank[count, :] = set_nframes_to_fixed_size(mfcc.sig2logspec(aud[int(start*16000):int(end*16000)]).T, 100)
            cur_penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
            penult[count, :] = np.squeeze(cur_penult)
            count += 1 
      print(count) 
    else:
      if DEBUG:
        print('File Do Not Exist')
      continue
    if DEBUG:
      print(fbank.shape)    
   
    # Store the features to the current group; ignore those without known concepts
    if np.sum(np.sum(lbl[:, :nc], axis=1), axis=0) == 0:
      print('Not enough labels!')
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
  h5_feat = h5py.File('flickr_sent_fbank_penult_segmented2.h5', 'w')

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
