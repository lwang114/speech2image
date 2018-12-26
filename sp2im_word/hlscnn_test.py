import numpy as np
import json
import tflearn
from hlscnn_train import *

DEBUG = False
def convert_retrieval_res(res_file, id2wrd_file, readable_file):
  max_ids = np.loadtxt(res_file).tolist()
  with open(id2wrd_file, 'r') as f:
    id2wrd = f.read().strip().split('\n')
  
  readable = []
  for max_id_row in max_ids:
    readable_row = []
    for max_id in max_id_row:
      readable_row.append(id2wrd[int(max_id)])

    readable.append(' '.join(readable_row))
  with open(readable_file, 'w') as f:
    f.write('\n'.join(readable))

def recall_op_sentence(s_predict, y_gt, ntop, save_file=None, binary_error=False):
  max_k_ids = retrieve(s_predict, ntop)
  if DEBUG:
    print(max_k_ids.T.shape)
  if binary_error:
    max_lbls = [[np.min(y_gt[max_id] == y_gt[i]) for max_id in max_k_id_row] for i, max_k_id_row in enumerate(max_k_ids.T.tolist())]
#[[(np.sum(y_gt[max_id] * y_gt[i]) == np.sum(y_gt[i])) for max_id in max_k_id_row] for i, max_k_id_row in enumerate(max_k_ids.T.tolist())] 

  else:
    max_lbls = [[np.sum(y_gt[max_id] * y_gt[i])/np.sum(y_gt[i]) for max_id in max_k_id_row] for i, max_k_id_row in enumerate(max_k_ids.T.tolist())] 
  if save_file:
    ntx = len(y_gt)
    np.savetxt(save_file, np.concatenate([np.expand_dims(np.arange(ntx), axis=1), max_k_ids.T], axis=1))
  return np.mean(np.max(max_lbls, axis=1))

def classify_op_sentence(y_true, y_pred, concept_file='data/concepts.txt', savefile=True):
  with open(concept_file, 'r') as f:
    concept2id = f.read().strip().split('\n')
  with open('concept_classify_res.txt', 'w') as f:
    for y_t, y_p in zip(y_true.tolist(), y_pred.tolist()):
      txt_true = []
      txt_pred = []
      for atom_t in y_t:
        if DEBUG:
          print(atom_t)
        txt_true.append(concept2id[atom_t])
      f.write(' '.join(txt_true) + '\n')
      
      for atom_p in y_p:
        txt_pred.append(concept2id[atom_p])
      f.write(' '.join(txt_pred) + '\n\n')
  print(y_true.shape)
  return np.mean(y_true == y_pred)

def test(data_file, model_file, nclass, hinge_loss=False, sentence=False, n_word_obj = 5, concept_file='data/concepts.txt'):
  # Load the data from h5 file
  h5_feat = h5py.File(data_file)
  grp_feat_tx = h5_feat['test']

  # Initialize the model
  g1 = tf.Graph()
  with g1.as_default():
    model_sp2im = HLSCNN(nclass=nclass, n_word_obj=n_word_obj)
    model_sp2im.load(model_file)

  # Put the h5 file into matrix format
  with open(concept_file, 'r') as f:
    concepts = f.read().strip().split('\n')
  nconcept = len(concepts)

  if sentence:
    a_feat_tx = np.array(grp_feat_tx['fbank'])
    v_feat_tx = np.array(grp_feat_tx['vgg_penult'])
    y_tx = np.array(grp_feat_tx['lbl'])
    if DEBUG:
      print(a_feat_tx.shape, v_feat_tx.shape, y_tx.shape)  
  else:
    tx_item = [it for c, grp in grp_feat_tx.items() for i, it in enumerate(grp.items()) if it[0] and i < 10 and c in concepts]
  
    tx_item_sorted = sorted(tx_item, key=lambda x: int(x[0].split('_')[-1]))
    tx_list = [d for k, d in tx_item_sorted]
    wrd_list = ['_'.join(k.split('_')[-2:]) for k, d in tx_item_sorted] 
    with open('ind2wrd_test.txt', 'w') as f:
      f.write('\n'.join(wrd_list))  

    a_feat_tx = [dset['fbank'] for dset in tx_list] 
    a_feat_tx = np.array(a_feat_tx)
    v_feat_tx = [dset['vgg_penult'] for dset in tx_list]
    v_feat_tx = np.array(v_feat_tx)
    y_tx = [dset['concept_lbl'] for dset in tx_list]
    y_tx = np.array(y_tx)
    if DEBUG:    
      print(y_tx.shape)
  
  if DEBUG:
    print(a_feat_tx[:, np.newaxis, :, :].shape)
  a_feat_tx = np.transpose(a_feat_tx, [0, 1, 3, 2]) 
 
  # Store the concepts predicted by the classifiers in the network 
  concept_preds = []
  if sentence:
    a_feat_tx_sample, v_feat_tx_sample, y_tx_sample = select_retrieval_database(a_feat_tx, v_feat_tx, y_tx, 'data/flickr_sentence_segment/ind2sent_test_cleanup.txt', 'data/flickr_sentence_segment/im2sent_test.json')
    ntx = a_feat_tx_sample.shape[0]
    batch_size = 32
    nbatch = int(ntx/batch_size) 
    similarity = np.zeros((ntx, ntx))
    if not hinge_loss:
      for i in range(nbatch):
        a_batch = a_feat_tx_sample[i*batch_size:(i+1)*batch_size]
        for j in range(nbatch):
          print('Audio batch {}, visual batch {}'.format(i, j)) 
          v_batch = v_feat_tx_sample[i*batch_size:(i+1)*batch_size]
          comb_vec = model_sp2im.predict([a_batch, v_batch])
          av_vec = comb_vec[:, :nconcept]
          a_vec = comb_vec[:, nconcept:] 
          if DEBUG:
            print(a_vec.shape)
            print(av_vec.shape)
          similarity[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = np.dot(a_vec, av_vec.T)         
    else:
      for i in range(nbatch):
        a_batch = a_feat_tx_sample[i*batch_size:(i+1)*batch_size]
        for j in range(nbatch):
          print('Audio batch {}, visual batch {}'.format(i, j)) 
          v_batch = v_feat_tx_sample[i*batch_size:(i+1)*batch_size]
          comb_vec = model_sp2im.predict([a_batch, v_batch])
          if DEBUG:
            print(comb_vec.shape)
          if i == j:
            y_preds_onehot = comb_vec[:, :-batch_size].reshape(batch_size, n_word_obj, nconcept)
            y_preds = np.argmax(y_preds_onehot, axis=2)
            concept_preds.append(y_preds)
          s = comb_vec[:, (-1-batch_size):-1]
          similarity[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = s         
      
      concept_preds = np.concatenate(concept_preds, axis=0)
      
  else:
    similarity = []
    for a_feat in a_feat_tx.tolist():
      #a_feats = []
      #v_feats = []
      s_predict_sp2im = []
      for v_feat in v_feat_tx.tolist():
        #a_feats.append(a_feat)
        #v_feats.append(v_feat)
        y_predict_sp2im = model_sp2im.predict([[np.array(a_feat)], [np.array(v_feat)]])
        y_predict_sp = model_sp.predict([a_feat])
        c = np.argmax(y_predict_sp)
        s_predict_sp2im.append(np.squeeze(y_predict_sp2im[:, c]))
      similarity.append(s_predict_sp2im)

  similarity = np.array(similarity)
  concept_preds = np.array(concept_preds)
  if DEBUG:
    print(similarity.shape)
    
  if sentence:
    #print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 1, 'recall_at_1_sent.txt'))
    #print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 5, 'recall_at_5_sent.txt'))
    #print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 10, 'recall_at_10_sent.txt'))
    print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 1, 'recall_at_1_sent.txt', binary_error=True))
    print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 5, 'recall_at_5_sent.txt', binary_error=True))
    print(recall_op_sentence(similarity, np.sum(y_tx_sample, axis=1), 10, 'recall_at_10_sent.txt', binary_error=True))
    if hinge_loss:
      # Discard labels not used in the minibatch testing due to the fixed minibatch constraint 
      print(classify_op_sentence(np.argmax(y_tx_sample[:concept_preds.shape[0]], axis=2), concept_preds, concept_file=concept_file))    
  else:
    print(recall_op_concept(similarity, y_tx, 1, 'recall_at_1_res.txt'))
    print(recall_op_concept(similarity, y_tx, 5, 'recall_at_5_res.txt'))
    print(recall_op_concept(similarity, y_tx, 10, 'recall_at_10_res.txt'))

if __name__ == '__main__':
  #clscnn_test('/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_wrd_fbank_penult_70concepts.h5', '/home/lwang114/spring2018/sp2im_word/sp2im_wrd_model_tflearn-18-06-07-21', '/home/lwang114/spring2018/sp2im_word/sp_wrd_model_tflearn-18-06-07-21', 70)
  #clscnn_test('/home/lwang114/spring2018/sp2im_word/data/flickr_sent_fbank_penult_order.h5', '/home/lwang114/spring2018/sp2im_word/models/sp2im_wrd_model_tflearn-18-07-08-09', 70, sentence=True)

  test('/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment/flickr_sent_fbank_penult_segmented.h5', '/home/lwang114/spring2018/sp2im_word/sp2im_wrd_model_tflearn-18-07-29-12', 71, hinge_loss=True, sentence=True, concept_file='data/concepts.txt')
  #convert_retrieval_res('recall_at_10_res.txt', 'ind2wrd_test.txt', 'recall_at_10_res_readable.txt')
