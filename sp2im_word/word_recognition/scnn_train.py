############################# scnn_train.py ################################
# This is the Spectrogram CNN model for pretraining of the speech-to-image 
# model using word segmented speech described by Harwath & Glass (2015) 
# paper.
#   
# Liming Wang, Mar. 26th
#
###########################################################################

import tensorflow as tf
import tflearn
#from keras.models import Sequential
#from keras.layers import Dense
import time
import glob
import numpy as np


########################### Helper Functions ##############################
DEBUG = True
def weight_variable(dims, init_method=None):
  '''A convenience function that creates a weight matrix,
     this is used by lscnn.scnn() and lscnn.vggembed()'''
  if init_method == "glorot_init":
     if len(dims) == 4:
       w = tf.Variable(tf.truncate_normal(dims, stddev=np.sqrt(2/(dims[0]*dims[1]*dims[2] + dims[0]*dims[1]*dims[3]))))
     else:
       w = tf.Variable(tf.truncate_normal(dims, stddev=np.sqrt(2/(dims[0] + dims[1]))))
  else:
     w = tf.Variable(tf.truncate_normal(dims, mean=0, stddev=0.01))
  return w

def bias_variable(dims):
  b = tf.Variable(tf.truncate_normal(dims, stddev=0.1))
  return b

def convert_to_one_hot(Y, nclass=None):
  nex = len(Y)
  Y_one_hot = np.zeros((nex, nclass))
  for i in range(nex):
    Y_one_hot[i, int(Y[i])] = 1 
  return Y_one_hot

######################### SCNN Network and Training ######################
def SCNN(nclass, classify=True, train_softmax=True, sentence=False, scope='word_recognizer', n_word=1):
   tf.reset_default_graph()
   if sentence:
    net = tflearn.input_data(shape=[None, n_word, 1024, 40])
   else:
    net = tflearn.input_data(shape=[None, n_word, 100, 40])
   
   net = tflearn.conv_2d(net, 64, [1, 5], padding='valid', activation='relu', bias=True, name='conv1', scope=scope+'/conv1', regularizer='L2')   
   net = tflearn.max_pool_2d(net, [1, 2], name='pool1')
   net = tflearn.conv_2d(net, 64, [1, 5], padding='valid', activation='relu', bias=True, name='conv2', scope=scope+'/conv2', regularizer='L2')   
   net = tflearn.max_pool_2d(net, [1, 2], name='pool2')

   if n_word > 1:
    net_list = tf.split(net, n_word, axis=1)
    net = tflearn.fully_connected(net_list[0], 512, bias=True, regularizer='L2', name='fc1', scope=scope+'/fc1')
    net = tf.stack(net + [ tflearn.fully_connected(net1, 512, bias=True, regularizer='L2', name='fc1', scope=scope+'/fc1', reuse=True) for net1 in net_list[1:] ], axis=1)
   else: 
    net = tflearn.fully_connected(net, 512, bias=True, regularizer='L2', name='fc1', scope=scope+'/fc1')
   
   net_embed = tflearn.dropout(net, 0.5, name='dropout1')
   if not classify:
    return net_embed
   if train_softmax:
     net = tflearn.fully_connected(net, nclass, activation='softmax')
   else:
     net = tflearn.fully_connected(net, nclass, activation='softmax', restore=False)
   
   if sentence:
    accuracy = tflearn.metrics.accuracy_multihot(nhot_max=3)
   else:
    accuracy = tflearn.metrics.weighted_accuracy()
   net = tflearn.regression(net, optimizer='adam', metric=accuracy, learning_rate=1e-5, loss='weighted_categorical_crossentropy')
    
   return net, net_embed

# Train the SCNN on a dataset specified by the paths 
def scnn_train(nclass, nword, tr_fbank_file='/home/lwang114/data/wsj_fbank_distinct10/wsj_fbank_distinct10_all.npz', tr_lbl_file='/home/lwang114/data/wsj_fbank_distinct10/wsj_lbl_distinct10.txt', model_file=None):
  # Initialize the Spectrogram CNN network model
  tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
     
  net, net_embed = SCNN(nclass=nclass)
  model = tflearn.DNN(net)
  model_embed = tflearn.DNN(net_embed)

  if model_file:
    model.load(model_file)
  # Load data
  f2X = np.load(tr_fbank_file)
  fbank_files = sorted(f2X, key=lambda x:int(x.split('_')[-1]))
  if nword == 1:
    X = np.array([np.expand_dims(f2X[f].transpose(), 0) for f in fbank_files])
    if DEBUG:
      print(X.shape)
    with open(tr_lbl_file, 'r') as f:
      Y_str = f.read().strip().split('\n')

    Y = [int(y.strip().split(' ')[-1]) for y in Y_str]
    Y_onehot = convert_to_one_hot(Y, nclass=nclass)
    if DEBUG:
      print(Y.shape) 
  else:
    X_1d = np.array([f2X[f].transpose() for f in fbank_files])
    nsample = X_1d.shape[0]
    nf = X_1d.shape[2]
    nt = X_1d.shape[3]
    ntr = int(nsample/nword)
    X = np.reshape(X_1d[:ntr*nword], [ntr, nword, nf, nt])
 
    if DEBUG:
      print(X.shape)
    with open(tr_lbl_file, 'r') as f:
      Y_str = f.read().strip().split('\n')

    Y = [int(y.strip().split(' ')[-1]) for y in Y_str]
    Y_onehot_1d = convert_to_one_hot(Y[:ntr*nword], nclass=nclass)
    Y_onehot = Y_onehot_1d.reshape([ntr, nword, nclass])
    if DEBUG:
      print(Y.shape)    

    tr_inds = np.random.randint(0, X.shape[0]-1, ntr)
    val_inds = np.array([i for i in range(ntr) if not i in tr_inds.tolist()])
    X_tr = X[tr_inds, :, :, :]
    X_val = X[val_inds, :, :, :]
    Y_onehot = convert_to_one_hot(lbls, nclass=nclass)
    Y_tr = Y_onehot[tr_inds, :]
    Y_val = Y_onehot[val_inds, :]

  # Split the data
  model.fit(X_tr, Y_tr, n_epoch=20, batch_size=32, shuffle=True, validation_set=(X_val, Y_val), show_metric=True)
  
  model.save('scnn_model_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))
  model.save('scnn_model_embed_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))

# Train the SCNN on a dataset as ndarrays
# nword specify the number of words used in one forward pass (useful for pretraining CLSCNN & HLSCNN)
def scnn_train2(fbanks, lbls, nword=10, model_file=None):
  # Initialize the Spectrogram CNN network model
  tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
  nclass = np.unique(lbls).shape[0]
  net, net_embed = SCNN(nclass, n_word=nword)
  model = tflearn.DNN(net)

  Y_onehot = convert_to_one_hot(lbls, nclass=nclass)

  if nword == 1:
    X = np.array([np.expand_dims(x.transpose(), 0) for x in fbanks])
    ntr = int(X.shape[0]*0.8)
    if DEBUG:
      print(X.shape)
       
  else:
    X = np.array([x.transpose() for x in fbanks])
    nsample = X.shape[0]
    nt = X.shape[1]
    nf = X.shape[2]
    ntr = int(nsample/nword)
    X = np.reshape(X[:ntr*nword], [ntr, nword, nt, nf])
 
    if DEBUG:
      print(X.shape)
    
    Y_onehot = Y_onehot[:ntr*nword].reshape([ntr, nword, nclass])
    if DEBUG:
      print(Y_onehot.shape)    

  if model_file:
    model.load(model_file)
  # Load data
  #X = np.array([np.expand_dims(x.transpose(), 0) for x in fbanks])
  ntr = int(X.shape[0]*0.8)
  tr_inds = np.random.randint(0, X.shape[0]-1, ntr)
  val_inds = np.array([i for i in range(ntr) if not i in tr_inds.tolist()])
  X_tr = X[tr_inds, :, :, :]
  X_val = X[val_inds, :, :, :]
  #Y_onehot = convert_to_one_hot(lbls, nclass=nclass)
  Y_tr = Y_onehot[tr_inds, :]
  Y_val = Y_onehot[val_inds, :]

  # Split the data
  model.fit(X_tr, Y_tr, n_epoch=20, batch_size=32, shuffle=True, validation_set=(X_val, Y_val), show_metric=True)
  
  model.save('scnn_model_tflearn{}'.format(time.strftime("-%y-%m-%d-%H", time.localtime())))
 

if __name__ == "__main__":
  #model_files = glob.glob("model/scnn_model_tflearn*.data-00000-of-00001")
  #time_scores = []
  #for f in model_files:
  #  dates = f.split('.')[0].split('-')[-4:-1]
  #  s = sum([int(d) for d in dates])
  #  time_scores.append(s)
  
  #model_file = model_files[int(np.argsort(-np.array(time_scores))[0])].split('.')[0]
  #print(model_file)
  #scnn_train(nclass=10, model_file=model_file)
  scnn_train(10, 5)
