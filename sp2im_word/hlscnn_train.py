########################### hlscnn_train.py #######################################
# This code implements the training procedure for the speech-to-image 
# retrival model called Hierarchical Latent Spectrogram Convolutional Neural
# Net (HLSCNN). To run the code, 
# simply modify the data paths for the data and pretrain models to the 
# right paths, copy and paste the hg16_hinge_loss function to 
# tflearn/objective.py and run the command 'python train.py' on the 
# terminal. 
#
# Author: Liming Wang
# Date created: July. 22nd, 2018 
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
  if len(a_vec.get_shape()) > 2:
    a_norm = a_vec / tf.norm(a_vec, ord=2, axis=0)
    v_norm = v_vec / tf.norm(v_vec, ord=2, axis=0)
    n = a_norm.get_shape()[1]
    a_list = tf.split(a_norm, n, axis=1)
    a_list = [tf.squeeze(a, axis=1) for a in a_list]
    v_list = tf.split(v_norm, n, axis=1)
    v_list = [tf.squeeze(v, axis=1) for v in v_list]
    if DEBUG:
      print(len(a_list), len(v_list))
      print(a_list[0].shape)
      print('Line 37', tf.matmul(a_list[0], v_list[0], transpose_b=True).shape)
    s_list = tf.stack([tf.stack([tf.nn.relu(tf.matmul(a, v, transpose_b=True)) for v in v_list], axis=0) for a in a_list], axis=0)
    s = tf.reduce_sum(tf.reduce_sum(s_list, axis=1), axis=0)
    if DEBUG:
      print('Line 40', s.shape)
  else:
    a_norm = tf.nn.l2_normalize(a_vec, dim=0)
    v_norm = tf.nn.l2_normalize(v_vec, dim=0)
    s =  tf.nn.relu(tf.matmul(a_norm, v_norm, transpose_b=True))
  return s

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
    #if DEBUG:
    #print(sent)
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

def image_encoder(scopes=[None, None], reuse=False, n_obj=1):
  if n_obj > 1:
    net = tflearn.input_data(shape=[None, n_obj, 4096], name=scopes[0])
    net_list = tf.split(net, n_obj, axis=1)
    net = tf.stack([tflearn.fully_connected(net1, 512, bias=True, reuse=reuse) for net1 in net_list], axis=1, name=scopes[1])
  else:
    net = tflearn.input_data(shape=[None, 4096], name=scopes[0])
    net = tflearn.fully_connected(net, 512, bias=True, scope=scopes[1], reuse=reuse)
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

# Hierarchical Latent SCNN
def HLSCNN(nclass, weight_file=None, n_word_obj=5, pretrain=False):
  sp_encs = SCNN(0, classify=False, sentence=False, n_word=n_word_obj)
  im_enc = image_encoder(scopes=['in_pos', 'fc_im'], n_obj=n_word_obj)
  if DEBUG:
    print(sp_enc.get_shape(), im_enc.get_shape())
  comb_enc = sp_enc * im_enc 
  comb_enc_list = tf.split(comb_enc, n_word_obj, axis=1)
  w_cls = tf.Variable(tf.truncated_normal((sp_enc.get_shape()[-1], nclass), stddev=0.01))
  b_cls = tf.Variable(tf.truncated_normal((ncls,), stddev=0.01))
  net_cls_list = tf.concatenate([ tf.nn.softmax(tf.matmul(enc, w_cls) + b_cls) for enc in comb_enc_list ], axis=1) 
  net_ret = cosine_similarity(sp_enc, im_enc)
 
  # Define the classification layer for sp2im retriever
  accuracy = tflearn.metrics.weighted_accuracy() 
  reg_cls = tflearn.regression(net_cls, optimizer='adam', metric=accuracy, learning_rate=1e-5, loss='weighted_categorical_crossentropy') 
  model_cls = tflearn.DNN(reg_cls, tensorboard_dir='./experiments/{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))

  if pretrain:
    return model_cls

  # Define the retrieval layer for sp2im retriever
  recall = tflearn.metrics.recall(n=1)
  reg_ret = tflearn.regression(net_ret, optimizer='adam', metric=recall, learning_rate=1e-5, loss='hg16_hinge_loss')
  if DEBUG:
    print('Line 153', len(reg_cls_list))
    print('Line 154', reg_cls_list[0].shape)
    print('Line 155', reg_ret.shape)
  
  # Combine the classification and retrieval layers
  net = tflearn.merge([reg_cls, reg_ret], mode='concat', axis=1)  
  model_comb = tflearn.DNN(net, tensorboard_dir='./experiments/{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))
  return model_comb, model_cls

# Training function
def train(data_file, nclass, max_margin=False, hinge_loss=False, sp2im_model=None, n_word_obj=5, pretrain_model=None, pretrain=False, sp_model=None):
  # Load the data from h5 file
  h5_feat = h5py.File(data_file)
  data_dir = '/'.join(data_file.split('/')[:-1])

  # split into training and testing set
  grp_feat_tr = h5_feat['train']
  grp_feat_tx = h5_feat['test']
  grp_feat_val = h5_feat['val']

  grps_tr = grp_feat_tr.items()
  
  a_feat_tr = np.array(grp_feat_tr['fbank'])
  v_feat_tr = np.array(grp_feat_tr['vgg_penult'])
  y_tr = np.array(grp_feat_tr['lbl'])
    
  if DEBUG:
    print('Line 176', a_feat_tr.shape, v_feat_tr.shape, y_tr.shape)

  a_feat_tx = np.array(grp_feat_tx['fbank'])
  v_feat_tx = np.array(grp_feat_tx['vgg_penult'])
  y_tx = np.array(grp_feat_tx['lbl'])

  a_feat_val = np.array(grp_feat_val['fbank'])
  v_feat_val = np.array(grp_feat_val['vgg_penult'])
  y_val = np.array(grp_feat_val['lbl'])

  # Change the features to the right shape
  if DEBUG:
    print('Line 188', a_feat_tx.shape, v_feat_tx.shape, y_tx.shape)
  a_feat_tr = np.transpose(a_feat_tr, [0, 1, 3, 2])
  a_feat_tx = np.transpose(a_feat_tx, [0, 1, 3, 2])
  a_feat_val = np.transpose(a_feat_val, [0, 1, 3, 2])
 
  # Initialize the sp2im retriever
  tf.reset_default_graph()
  g1 = tf.Graph()
  # Train for 20 epochs
  nep = 20
  batch_size = 128
  ntr = a_feat_tr.shape[0]
 
  with g1.as_default():
    if pretrain:
      model_pretrain = HLSCNN(nclass=nclass, n_word_obj=n_word_obj, pretrain=True, sp_model=sp_model)
      #y_tr_tile = [np.tile(y_tr, n_word_obj) for y in y_tr.tolist()]
      #y_tr_tile = np.array(y_tr_tile)
      #y_val_tile = [np.tile(y_val, n_word_obj) for y in y_val.tolist()]
      #y_val_tile = np.array(y_val_tile)
      model_pretrain.fit([a_feat_tr, v_feat_tr], y_tr, n_epoch=nep, batch_size=batch_size, validation_set=([a_feat_val, v_feat_val], y_val_tile), shuffle=True, show_metric=True)
      model_pretrain.save('models/sp2im_cls_model_tflearn{}'.format(time.strftime("%y-%m-%d-%H", time.localtime())))
    else:
      model_sp2im, model_pretrain = HLSCNN(nclass=nclass, n_word_obj=n_word_obj)
    if pretrain_model:
      model_pretrain.load(pretrain_model) 
      
    elif sp2im_model:
      model_sp2im.load(sp2im_model)  
    for i in range(1):
      a_feat_tr_sample, v_feat_tr_sample, y_tr_sample = select_retrieval_database(a_feat_tr, v_feat_tr, y_tr, ind2sent_file=data_dir+'/ind2sent_train_cleanup.txt', im2sent_file= data_dir+'/im2sent_train.json', random=True)
      a_feat_val_sample, v_feat_val_sample, y_val_sample = select_retrieval_database(a_feat_val, v_feat_val, y_val, ind2sent_file=data_dir+'/ind2sent_val_cleanup.txt', im2sent_file=data_dir+'/im2sent_val.json', random=True) 
      y_tr_sample_list = np.split(y_tr_sample, n_word_obj, axis=1)
      y_tr_sample_list = [np.squeeze(y_tr, 1) for y_tr in y_tr_sample_list]
      # Placeholder for the retrieval loss
      y_tr_sample_list.append(np.zeros((y_tr_sample.shape[0], 1)))
      y_val_sample_list = np.split(y_val_sample, n_word_obj, axis=1)
      y_val_sample_list = [np.squeeze(y_val, 1) for y_val in y_val_sample_list]
      # Placeholder for the retrieval loss
      y_val_sample_list.append(np.zeros((y_tr_sample.shape[0], 1)))
 
      if DEBUG:
        print('Line 220', a_feat_tr_sample.shape, v_feat_tr_sample.shape, y_tr_sample.shape)
      model_sp2im.fit([a_feat_tr_sample, v_feat_tr_sample], y_tr_sample_list, n_epoch=nep, batch_size=batch_size, validation_set=([a_feat_val_sample, v_feat_val_sample], y_val_sample_list), shuffle=True, show_metric=True)
      model_sp2im.save('models/sp2im_ret_model_tflearn{}'.format(time.strftime("%y-%m-%d-%H", time.localtime())))
        
if __name__ == "__main__":
  #train('/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_wrd_fbank_penult_70concepts.h5', nclass=70, max_margin=False)
  #train('/home/lwang114/spring2018/sp2im_word/data/flickr_sent_fbank_penult_order.h5', nclass=70, hinge_loss=True, sentence=True, sp2im_model='/home/lwang114/spring2018/sp2im_word/sp2im_wrd_model_tflearn-18-07-15-17')     
  train('/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment/flickr_sent_fbank_penult_segmented2.h5', nclass=71, hinge_loss=True) #sp2im_model='/home/lwang114/spring2018/sp2im_word/models/sp2im_wrd_model_tflearn-18-08-08-09')
