########################### train.py #######################################
# This code implements the training procedure for the speech-to-image 
# retrival model for spoken words and visual objects. To run the code, 
# simply modify the data paths for the data and pretrain models to the 
# right paths, copy and paste the hg16_hinge_loss function to 
# tflearn/objective.py and run the command 'python train.py' on the 
# terminal. 
#
# Author: Liming Wang
# Date created: May. 23rd, 2018 
############################################################################

import numpy as np
import h5py
import tensorflow as tf
import tflearn
import copy
import json
from word_recognition.scnn_train import *
from image_grounding.image_encoder_pretrain import *

DEBUG = False
def cosine_similarity(a_vec, v_vec):
  a_norm = tf.nn.l2_normalize(a_vec, dim=0)
  v_norm = tf.nn.l2_normalize(v_vec, dim=0)
  return tf.nn.relu(tf.matmul(a_norm, v_norm, transpose_b=True))

# Hinge loss function used by Harwath and Glass 2016; y_true is used only to match the pattern of tflearn objective function 
'''def hg16_hinge_loss(y_pred, y_true):
  with tf.name_scope(None):
    return tf.reduce_mean(tf.nn.relu(y_pred - tf.diag(y_pred) + 1)) + tf.reduce_sum(tf.nn.relu(tf.transpose(y_pred) - tf.diag(y_pred) + 1))
'''
def select_retrieval_database(a_feat, v_feat, y, ind2sent_file, im2sent_file, random=False):
  with open(im2sent_file, 'r') as f:
    im2sent = json.load(f)
  with open(ind2sent_file, 'r') as f:
    ind2sent = f.read().strip().split('\n')
  sent2ind = {s:i for i, s in enumerate(ind2sent)}

  selected_indices = []
  for im in im2sent.keys():
    if random:
      n_tokens = len(im2sent[im])
      rand_ind = np.random.randint(0, n_tokens) 
      sent = im2sent[im][rand_ind]
    else:
      sent = im2sent[im][0]
    selected_indices.append(sent2ind[sent])
    
  np.savetxt('experiments/dbid2id_test.txt', selected_indices)
  if DEBUG:
    print(a_feat.shape)
    print(len(selected_indices))
  return a_feat[selected_indices], v_feat[selected_indices], y[selected_indices] 

def select_negative_examples(grp_list_tr, random=True):
  neg_vgg = []
  neg_fbank = []
  concepts = [c for c, grp in grp_list_tr]
  nc = len(concepts)
  for c, grp in enumerate(grp_list_tr):
    nk = len(grp.items())
    # Select negative features
    cneg2ind = {c2:i for i,c2 in enumerate(concepts) if not c2 == c}
    cneg = [c2 for c2 in concepts if not c2 == c]
    inds_neg = np.random.randint(low=0, high=nc-1, size=(nk,))
    for i in inds_neg:
       grp_neg = grp_list_tr[cneg2ind[cneg[i]]][1]
       keys_neg = sorted(grp_neg.keys())
       nk_neg = len(keys_neg)
       ind_key_neg = np.random.randint(low=0, high=nk_neg-1)
       neg_vgg.append(grp_neg[keys_neg[ind_key_neg]]['vgg_penult'])
       neg_fbank.append(grp_neg[keys_neg[ind_key_neg]]['fbank'])
  return np.array(neg_vgg), np.array(neg_fbank)

def image_encoder(scopes=[None, None], reuse=False):
  net = tflearn.input_data(shape=[None, 4096], name=scopes[0])
  net = tflearn.fully_connected(net, 512, scope=scopes[1], reuse=reuse)
  return net

def retrieve(s_predict, ntop):
  dummy = -2
  s = copy.deepcopy(s_predict)
  n = s.shape[0]
  max_k_ids = np.zeros((ntop, n), dtype=int)
  for i in range(ntop):
    max_ids = np.argmax(s, axis=1)
    max_k_ids[i, :] = max_ids
    for j in range(n):
      s[j, int(max_ids[j])] = dummy  
  return max_k_ids 

def recall_op(s_predict, ntop, save_file=None):
  # Calculate the recall score by Finding the top k elements of every column of the similarity matrix in linear time
  max_k_ids = retrieve(s_predict, ntop)
  dev = max_k_ids - np.linspace(0, n-1, n)
  if save_file:
    np.savetxt(save_file, max_k_ids.T)
  return np.mean(np.min(np.abs(dev), axis=0)==0)

def recall_op_concept(s_predict, y_gt, ntop, save_file=None):
  max_k_ids = retrieve(s_predict, ntop)
  if DEBUG:
    print(max_k_ids.T.shape)
  max_lbls = [[np.argmax(y_gt[max_id]) for max_id in max_k_id_row] for max_k_id_row in max_k_ids.T.tolist()] 
  dev = np.array(max_lbls).T - np.argmax(y_gt, axis=1)
  if save_file:
    ntx = len(y_gt)
    np.savetxt(save_file, np.concatenate([np.expand_dims(np.arange(ntx), axis=1), max_k_ids.T], axis=1))
  return np.mean(np.min(np.abs(dev), axis=0)==0)

# Concept-based Latent SCNN
def CLSCNN(nclass, weight_file=None, max_margin=False, sentence=False):
  sp_enc = SCNN(0, classify=False, sentence=sentence)
  im_enc = image_encoder(scopes=['in_pos', 'fc_im'])
  comb_enc = sp_enc * im_enc 
  net1 = tflearn.fully_connected(comb_enc, nclass, activation='softmax', scope='out1')

  net2 = tflearn.fully_connected(sp_enc, nclass, activation='softmax', scope='out2')
  # Define the regression layer for sp2im retriever
  '''if max_margin:
    #accuracy = tflearn.metrics.accuracy
    im_enc_neg = image_encoder(scopes=['in_neg', 'fc_im'], reuse=True)
    comb_enc_neg = sp_enc * im_enc_neg
    net_neg = tflearn.fully_connected(comb_enc, nclass, activation='softmax', scope='out', reuse=True)
    net_comb = tflearn.regression([net, net_neg], optimizer='adam', metric=None, learning_rate=1e-5, loss='maxmargin_categorical_crossentropy')
  
  else:'''
  accuracy = tflearn.metrics.accuracy_multihot(nhot_max=5)
 
  reg1 = tflearn.regression(net1, optimizer='adam', metric=accuracy, learning_rate=1e-5, loss='categorical_crossentropy') 
  
  reg2 = tflearn.regression(net2, optimizer='adam', metric=accuracy, learning_rate=1e-5, loss='categorical_crossentropy')

  merge = tflearn.merge([reg1, reg2], mode='concat', axis=1)
  if DEBUG:
    print('Ok here line 106!')
 
  return tflearn.DNN(merge)

# Concept-based Latent SCNN with cross entropy + max margin loss 
def CLSCNN2(nclass, weight_file=None, sentence=True):
  sp_enc = SCNN(0, classify=False, sentence=sentence)
  im_enc = image_encoder(scopes=['in_pos', 'fc_im'])
  comb_enc = sp_enc * im_enc 
  
  net1 = tflearn.fully_connected(comb_enc, nclass, activation='sigmoid') 
  net2 = cosine_similarity(sp_enc, im_enc)
 
  # Define the regression layer for sp2im retriever
  accuracy = tflearn.metrics.accuracy_multihot() 
  reg1 = tflearn.regression(net1, optimizer='adam', metric=accuracy, learning_rate=1e-5, loss='weighted_sigmoid_categorical_crossentropy') 
  recall = tflearn.metrics.recall(n=1)
  reg2 = tflearn.regression(net2, optimizer='adam', metric=recall, learning_rate=1e-5, loss='hg16_hinge_loss')

  net = tflearn.merge([reg1, reg2], mode='concat', axis=1)  
  return tflearn.DNN(net, tensorboard_dir='./experiments/')

def train(data_file, nclass, max_margin=False, hinge_loss=False, sentence=False, sp2im_model=None):
  # Load the data from h5 file
  h5_feat = h5py.File(data_file)
  data_dir = '/'.join(data_file.split('/')[:-1]) + '/'
  # split into training and testing set
  grp_feat_tr = h5_feat['train']
  grp_feat_tx = h5_feat['test']
  grp_feat_val = h5_feat['val']

  grps_tr = grp_feat_tr.items()
  if sentence:
    a_feat_tr = np.array(grp_feat_tr['fbank'])
    v_feat_tr = np.array(grp_feat_tr['vgg_penult'])
    y_tr = np.array(grp_feat_tr['lbl'])
    
  else:
    tr_list = [d for c, grp in grps_tr for k, d in grp.items() if k]
    tr_list = tr_list[:64]
    a_feat_tr = np.array([dset['fbank'] for dset in tr_list]) 
    v_feat_tr = np.array([dset['vgg_penult'] for dset in tr_list])
    y_tr = np.array([dset['concept_lbl'] for dset in tr_list])
  if DEBUG:
    print(a_feat_tr.shape, v_feat_tr.shape, y_tr.shape)

  if sentence:
    a_feat_tx = np.array(grp_feat_tx['fbank'])
    v_feat_tx = np.array(grp_feat_tx['vgg_penult'])
    y_tx = np.array(grp_feat_tx['lbl'])
  else:
    grps_tx = grp_feat_tx.items()
    tx_list = [d for c, grp in grps_tx for k, d in grp.items() if k]
    tx_list = tx_list[:64]
    a_feat_tx = np.array([dset['fbank'] for dset in tx_list]) 
    v_feat_tx = np.array([dset['vgg_penult'] for dset in tx_list])
    y_tx = np.array([dset['concept_lbl'] for dset in tx_list])
     
  if sentence:
    a_feat_val = np.array(grp_feat_val['fbank'])
    v_feat_val = np.array(grp_feat_val['vgg_penult'])
    y_val = np.array(grp_feat_val['lbl'])

  else:
    grps_val = grp_feat_val.items()
    val_list = [d for c, grp in grps_val for k, d in grp.items() if k]
    val_list = val_list[:64]
    a_feat_val = np.array([dset['fbank'] for dset in val_list]) 
    v_feat_val = np.array([dset['vgg_penult'] for dset in val_list])
    y_val = np.array([dset['concept_lbl'] for dset in val_list])

  # Change the features to the right shape
  if DEBUG:
    print(a_feat_tx.shape, v_feat_tx.shape, y_tx.shape)
  a_feat_tr = np.transpose(np.expand_dims(a_feat_tr, 1), [0, 1, 3, 2])
  a_feat_tx = np.transpose(np.expand_dims(a_feat_tx, 1), [0, 1, 3, 2])
  a_feat_val = np.transpose(np.expand_dims(a_feat_val, 1), [0, 1, 3, 2])
 
  v_feat_tr = np.squeeze(v_feat_tr, axis=1)
  v_feat_tx = np.squeeze(v_feat_tx, axis=1)
  v_feat_val = np.squeeze(v_feat_val, axis=1) 

  # Initialize the sp2im retriever
  tf.reset_default_graph()
  g1 = tf.Graph()
  # Train for 20 epochs
  nep = 50
  batch_size = 128
  ntr = a_feat_tr.shape[0]
 
  with g1.as_default():
    if not max_margin:
      if not hinge_loss: 
        model_sp2im = CLSCNN(nclass=nclass, sentence=sentence)
      else:
        model_sp2im = CLSCNN2(nclass=nclass, sentence=sentence)

      if sp2im_model:
        model_sp2im.load(sp2im_model) 
      
      if not hinge_loss:
        model_sp2im.fit([a_feat_tr, v_feat_tr], [y_tr, y_tr], n_epoch=nep, batch_size=batch_size, validation_set=([a_feat_val, v_feat_val], [y_val, y_val]), shuffle=True, show_metric=True)
        model_sp2im.save('sp2im_wrd_model_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))
      else:
        for i in range(5):
          a_feat_tr_sample, v_feat_tr_sample, y_tr_sample = select_retrieval_database(a_feat_tr, v_feat_tr, y_tr, ind2sent_file=data_dir+'ind2sent_train_cleanup.txt', im2sent_file=data_dir+'im2sent_train.json', random=True)
          a_feat_val_sample, v_feat_val_sample, y_val_sample = select_retrieval_database(a_feat_val, v_feat_val, y_val, ind2sent_file=data_dir+'ind2sent_val_cleanup.txt', im2sent_file=data_dir+'im2sent_val.json', random=True) 
          if DEBUG:
            print(a_feat_tr_sample.shape, v_feat_tr_sample.shape, y_tr_sample.shape)
          model_sp2im.fit([a_feat_tr_sample, v_feat_tr_sample], [y_tr_sample, y_tr_sample], n_epoch=nep, batch_size=batch_size, validation_set=([a_feat_val_sample, v_feat_val_sample], [y_val_sample, y_val_sample]), shuffle=True, show_metric=True)
        model_sp2im.save('sp2im_wrd_model_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))
        
    else:
      model_sp2im = CLSCNN(nclass=nclass, max_margin=True)
      iv_feat_tr_neg, a_feat_tr_neg = select_negative_examples(grps_tr)
      v_feat_val_neg, v_feat_val_neg = select_negative_examples(grps_val)
   
      # Train for 20 epochs
      model_sp2im.fit([a_feat_tr, v_feat_tr, v_feat_tr_neg], y_tr, n_epoch=nep, batch_size=batch_size, validation_set=([a_feat_val, v_feat_val, v_feat_val_neg], y_val), shuffle=True, show_metric=False)
      model_sp2im.save('sp2im_wrd_model_maxmargin_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))

if __name__ == "__main__":
  #train('/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_wrd_fbank_penult_70concepts.h5', nclass=70, max_margin=False)
  #train('/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment_807_2018/flickr_sent_fbank_penult_segmented2.h5', nclass=70, hinge_loss=True, sentence=True) #sp2im_model='/home/lwang114/spring2018/sp2im_word/sp2im_wrd_model_tflearn-18-07-15-17')     
  train('/home/lwang114/spring2018/sp2im_word/data/flickr_sentence/flickr_sent_fbank_penult_order.h5', nclass=70, hinge_loss=True, sentence=True)
