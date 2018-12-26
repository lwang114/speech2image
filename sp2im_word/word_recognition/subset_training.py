########################## Subset Training ################################
# This code trains a neural network incrementally on subset of data 
# constantly expanding into a large dataset. It contains codes for 
# spliting the dataset into several subsets by specifying the split
# in .txt format. The subsets are then stored into an hdf5 file. 
#
# Liming Wang, Mar.27th, 2018
###########################################################################

import tensorflow as tf
import h5py
import tables
import json
import copy
from scnn_train import *

DEBUG = True
def create_word_dictionary(wordfiles):
     wrd_dict = dict()
     for wf in wordfiles:
         if len(wf.strip().split('_')) < 3:
             print('Wrong npy name format')
             continue
         wrd = wf.strip().split('_')[-2]
         if not wrd in wrd_dict.keys():
             wrd_dict[wrd] = [wf];
         else:
             wrd_dict[wrd].append(wf)
     with open('word2file_dict.json', 'w') as f:
         json.dump(wrd_dict, f)
     
def create_fbank_dataset(wsj_path, wrd2file, sel_wrds):
   a = np.load(wsj_path)
   #sel_wrds = ['thousand', 'securities', 'japanese', 'corporation', 'politic    al', 'operating', 'thirteen', 'twelve', 'business', 'annual']
   wrd2lbl = {w:i for i, w in enumerate(sel_wrds)}
  
   fbanks = [] 
   lbls = []
   wrdcount = 0
   for i, w in enumerate(sel_wrds):
     files_w = wrd2file[w]
     for j, f in enumerate(files_w):
       if DEBUG:
         print(f)
       fbank = a[f]
       if DEBUG:
         print(fbank.shape)
       #np.save(save_path_fbank+'fbank_{}_{}.npy'.format(w, str(wrdcount)), fb    ank)
       fbanks.append(fbank)
       wrdcount += 1
       lbls.append(wrd2lbl[w]) 
   #with open(save_path_lbl, 'w') as f:
   #  f.write('\n'.join(lbls))
   return fbanks, lbls

def combine_labels(Y1, Y2):
  nclass1 = np.unique(Y1).shape[0]
  nclass2 = np.unique(Y2).shape[0]
  if DEBUG:
    print('Number of new classes: ', nclass2)
  for y in Y2:
    Y1.append(y + nclass1)
  return Y1, nclass1 + nclass2 
  
def create_subsets_from_word_split(dataset_path, dict_path, subset_list_file):
  dict_path = "word2file_dict.json"
  h5file = h5py.File('wsj_distinct_word_split_4000wrds.h5', 'w')  
  with open(dict_path, 'r') as f:
    wrd2file = json.load(f)

  with open(subset_list_file, 'r') as f:
    subset_str = f.read().strip().split('\n')
  subset_list = []

  for s in subset_str:
    subset_list.append(s.strip().split(' '))
  #subset_list = subset_list[0:2]
  
  h5file.create_group('wrd_lbls')
  h5file.create_group('fbanks')
  for i, sel_words in enumerate(subset_list):
    X, Y = create_fbank_dataset(dataset_path, wrd2file, sel_words)
    dset_fbank = h5file['fbanks'].create_dataset('subset_{}'.format(str(i)), (len(X), 40, 100), maxshape=(None, 40, 100))  
    dset_wrd = h5file['wrd_lbls'].create_dataset('subset_{}'.format(str(i)), (len(Y),), maxshape=(None,))
    dset_fbank[...] = copy.deepcopy(X)
    dset_wrd[...] = copy.deepcopy(Y)
  
  h5file.close()

def subsets_train(subsets_file, model_file=None): 
  subsets = h5py.File(subsets_file, 'r+')
  skeys = subsets['fbanks'].keys()
  X_comb = []
  Y_comb = []
  step_size = 1
  increase_subset_size = False
  increment_size = 100
  for i, skey in enumerate(skeys):
    if increase_subset_size:
      n_subset_per_expand = int((i * step_size + 1)/increment_size)
    else:
      n_subset_per_expand = 1 
    X = list(np.array(subsets['fbanks/' + skey]))
    if DEBUG:
      print('Number of examples: ', len(X))
    Y = list(np.array(subsets['wrd_lbls/' + skey]))
    if DEBUG:
      print('Number of labels: ', len(Y))
    if X_comb == []:
      X_comb = X
      Y_comb = Y
      nclass = np.unique(Y_comb).shape[0]
    else:
      for x in X:
        X_comb.append(x)
      Y_comb, nclass = combine_labels(Y_comb, Y)
    if DEBUG:
      print('Number of distinct classes: ', nclass)
      print("Training on {} distinct words ...".format(str(nclass)))    
    if (i * step_size + 1) % increment_size == 0:
      #SCNN(nclass, train_softmax=False)
      scnn_train2(X_comb, Y_comb, model_file=model_file)

#def create_subsets_with_id_split():
#s_split = s.strip().split(' ')
#s_int = [int(i) for i in s_split]
    #subset_list.append(s_int)
 
if __name__ == "__main__":
  wsj_path = "/home/lwang114/data/wsj/wsj_fbank_train_long.npz"
  #fbank_npz = np.load(wsj_path)
  #fbank_files = fbank_npz.files
  #create_word_dictionary(fbank_files)
  #create_subsets_from_word_split(wsj_path, "word2file_dict.json", "wsj_goodwrds_split.txt")
  subsets_train("wsj_distinct_word_split_4000wrds.h5")  
