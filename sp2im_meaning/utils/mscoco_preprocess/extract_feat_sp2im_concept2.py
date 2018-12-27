# -*- coding: utf-8 -*-
import numpy as np
import h5py
import PIL
import scipy.io.wavfile
import scipy.misc
import cv2, numpy as np
from mfcc import *
from vgg16 import *
from pycocotools.coco import *
from SpeechCoco.speechcoco_API.speechcoco.speechcoco import *
from collections import defaultdict

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
  ins_file = "annotations/instances_val2014.json"
  wrd_obj_info_file = "mscoco_wrd_obj_info.json"
  sql_file = "val2014/val_2014.sqlite3"
  coco_api = COCO(ins_file)
  speech_api = SpeechCoco(sql_file)   
  # From documentation of MSCOCO, the number of categories is 80
  ncats = len(coco_api.cats)

  with open('concepts.json', 'r') as f:
    con2cat = json.load(f)  
  
  cat2lbl = defaultdict(int)
  if DEBUG:
    print(len(coco_api.cats.keys()))
  for i, cat in enumerate(coco_api.cats.values()):
    cat2lbl[cat['name']] = i
  
  with open('cat2lbl.json', 'w') as f:
    json.dump(cat2lbl, f)

  im_path = '/home/lwang114/data/mscoco/val2014/imgs/val2014/'
  aud_path = '/home/lwang114/data/mscoco/val2014/wav/'
  # Instantiate vgg model and mfcc class
  mfcc = MFCC()
  sess = tf.Session()
  imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
  vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

  with open(wrd_obj_info_file, 'r') as f:
    wrd_obj_pairs = json.load(f)
  
  # Create a dictionary to map the id of a file to its purpose in the karpathy split
  '''split_path_tr = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_train.npz'
  split_path_val = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_val.npz'
  split_path_tx = '/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr40k_speech_test.npz'
  npz_tr = np.load(split_path_tr)
  tr_files = npz_tr.keys() 
  npz_tx = np.load(split_path_tx)
  tx_files = npz_tx.keys()
  npz_val = np.load(split_path_val)
  val_files = npz_val.keys()
  '''
  idToPurpose = dict()

  #for f in tr_files:
  #  idToPurpose[f.split('/')[-1].split('_')[0]] = 'train'
  #for f in tx_files:
  #  idToPurpose[f.split('/')[-1].split('_')[0]] = 'test'
  for im_id in coco_api.imgToAnns.keys():
    idToPurpose[im_id] = 'val'

  # Generate fbank and penult features and store them in an h5 file; use a karpathy split
  nf = len(wrd_obj_pairs.keys())
  tr_count = 0
  tx_count = 0
  val_count = 0
  
  # Create a HDF-5 file  
  h5_feat = h5py.File('mscoco_wrd_fbank_penult_80concepts.h5', 'w')
  tr_grp = h5_feat.create_group('train')
  tx_grp = h5_feat.create_group('test')
  val_grp = h5_feat.create_group('val')
  
  for i in range(ncats):
    tr_grp.create_group(cat2lbl.keys()[i])
    tx_grp.create_group(cat2lbl.keys()[i])
    val_grp.create_group(cat2lbl.keys()[i])

  fbanks_tr = [[] for i in range(ncats)]
  penults_tr = [[] for i in range(ncats)]
  y_tr = [[] for i in range(ncats)]
  tr_files = []
  
  fbanks_tx = [[] for i in range(ncats)]
  penults_tx = [[] for i in range(ncats)]
  y_tx = [[] for i in range(ncats)]
  tx_files = []

  fbanks_val = [[] for i in range(ncats)]
  penults_val = [[] for i in range(ncats)]
  y_val = [[] for i in range(ncats)]
  val_files = []

  for pair_id, info in wrd_obj_pairs.items():
    aud_id, im_id, wrd = pair_id.split('_')[0], pair_id.split('_')[1], pair_id.split('_')[2]
    captions = speech_api.getImgCaptions(im_id)
    for capt in captions:
      if capt.captionID == aud_id:
        break
    aud_filename = capt.filename 
    aud_file = aud_path + aud_filename     
    print(aud_file, wrd) 
    
    #if im_id not in coco_api.imgs.keys():
    im_file = coco_api.loadImgs(int(im_id))[0]['file_name'] 
    #coco_api.download(im_path, [int(im_id)])
    im = np.array(PIL.Image.open(im_path + im_file))
    
    bb = info['bbox']
    #if DEBUG:
    #  print(info)
    timing = info['timechunk'] 
    
    y_onehot = np.zeros((ncats,))
    lbl = cat2lbl[con2cat[wrd]]
    y_onehot[lbl] = 1 
    
    # Extract fbank feature for the audio and create a group for the current word
    start = timing[0] / 1000
    end = timing[1] / 1000
    Fs, aud = scipy.io.wavfile.read(aud_file)
    #if DEBUG:
    #  print(aud[int(start*Fs):int(end*Fs)].shape)
    if DEBUG:
      print(im.shape) 
    fbank = set_nframes_to_fixed_size(mfcc.sig2logspec(aud[int(start*Fs):int(end*Fs)]).T, 100)
    #print(fbank.shape)    
    
    # Extract vgg penult feature for the visual objects
    try:
      reg = im[int(bb[1]):int(bb[1]+bb[3]), int(bb[0]):int(bb[0]+bb[2]), :]
    except:
      continue

    reg_fixsize = scipy.misc.imresize(reg, (224, 224, 3))
    penult = sess.run(vgg.fc1, feed_dict={vgg.imgs:[reg_fixsize]})
   
    # Store the features to the current group
    if idToPurpose[int(im_id)] == 'train':
      tr_files.append(' '.join([aud_id, im_id, wrd, str(start), str(end), 
      str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3])]))        
      fbanks_tr[lbl].append(fbank)
      penults_tr[lbl].append(penult)
      y_tr[lbl].append(y_onehot)
       
    elif idToPurpose[int(im_id)] == 'test':
      tx_files.append(' '.join([aud_id, im_id, wrd, str(start), str(end), 
      str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3])]))   
      fbanks_tx[lbl].append(fbank)
      penults_tx[lbl].append(penult)
      y_tx[lbl].append(y_onehot)
 
    else:
      val_files.append(' '.join([aud_id, im_id, wrd, str(start), str(end), 
      str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3])]))  
      fbanks_val[lbl].append(fbank)
      penults_val[lbl].append(penult)
      y_val[lbl].append(y_onehot)

  for i, c in enumerate(cat2lbl.keys()): 
    tr_grp[c].create_dataset('fbank', data=fbanks_tr[i])
    tr_grp[c].create_dataset('vgg_penult', data=penults_tr[i])
    tr_grp[c].create_dataset('concept_lbl', data=y_tr[i])
      
  with open('train_pairs.txt', 'w') as f:
    f.write('\n'.join(tr_files)) 
            
  for i, c in enumerate(cat2lbl.keys()): 
    tx_grp[c].create_dataset('fbank', data=fbanks_tx[i])
    tx_grp[c].create_dataset('vgg_penult', data=penults_tx[i])
    tx_grp[c].create_dataset('concept_lbl', data=y_tx[i])
      
  with open('test_pairs.txt', 'w') as f:
    f.write('\n'.join(tx_files))
      
  for i, c in enumerate(cat2lbl.keys()): 
    val_grp[c].create_dataset('fbank', data=fbanks_val[i])
    val_grp[c].create_dataset('vgg_penult', data=penults_val[i])
    val_grp[c].create_dataset('concept_lbl', data=y_val[i])

  with open('val_pairs.txt', 'w') as f:
    f.write('\n'.join(val_files)) 
