import numpy as np
import matplotlib.pyplot as plt
import h5py

DEBUG = True

def find_correct_retrieval(datafile, res_file, dbid2id_file, res_correct_file):
  h5file = h5py.File(datafile, 'r')
  y = np.array(h5file['test/lbl'])
  max_ids = np.loadtxt(res_file)
  dbid2id = np.loadtxt(dbid2id_file)

  global_max_ids = [[int(dbid2id[int(dbid)]) for dbid in dbid_row] for dbid_row in max_ids.tolist()]
  global_max_ids = np.array(global_max_ids)
  gt_ids = global_max_ids[:, 0]
  pred_ids = global_max_ids[:, 1:]
  if DEBUG:
    print(global_max_ids.shape)
    #print(np.sum(y[gt_ids[5]] * y[pred_ids[5, 3]]))
    #print(np.sum(y[gt_ids[5]]))
    #print(np.sum(y[pred_ids[5, 3]]))
  correct_ids = [[np.min(y[pred_id] == y[gt_id]) for pred_id in pred_ids[i].tolist()] for i, gt_id in enumerate(gt_ids.tolist())]
  correct_id = np.max(correct_ids, axis=1) 
  np.savetxt(res_correct_file, max_ids[correct_id])

def convert_retrieval_res_word(res_file, id2wrd_file, readable_file):
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

def convert_retrieval_res_sent(res_file, id2sent_file, dbid2id_file, readable_file):
  max_ids = np.loadtxt(res_file).tolist()
  if DEBUG:
    print(len(max_ids[0]))
  dbid2id = np.loadtxt(dbid2id_file) 

  with open(id2sent_file, 'r') as f:
    id2sent = f.read().strip().split('\n')
  readable = []
  concepts_list = [] 

  for i, max_id_row in enumerate(max_ids):
    readable_row = []
    for j, max_id in enumerate(max_id_row):
      sent = id2sent[int(dbid2id[int(max_id)])]
      readable_row.append(sent+'_'+str(int(dbid2id[int(max_id)])))

    readable.append(' '.join(readable_row))

  with open(readable_file, 'w') as f:
    f.write('\n'.join(readable))

def retrieval_res_concept_level(data_file, ret_sents_file, concept_file, concept_level_ret_res_file):
  h5file = h5py.File(data_file)
  y_tx = h5file['test/lbl']
  # For the case of segmented words 
  if len(y_tx.shape) > 2:
    y_tx = np.sum(y_tx, axis=1)
  if DEBUG:
    print('Line 70', y_tx.shape)

  with open(ret_sents_file, 'r') as f:
    ret_sents = f.read().strip().split('\n')
  
  with open(concept_file, 'r') as f:
    id2wrd = f.read().strip().split('\n')
  id2wrd.append('no-obj')
  concepts_info = []
  with open(concept_level_ret_res_file, 'w') as f:
    for i, sent_row in enumerate(ret_sents): 
      sents = sent_row.split(' ')
      f.write(sents[0] + '\n')
     
      for j, sent in enumerate(sents):
        spid = int(sent.split('_')[-1])  
        cur_y_pred = y_tx[spid]
        nc = int(np.sum(cur_y_pred))
        concept_ids_pred = [ k for k, y in enumerate(cur_y_pred) if y>=1 ]
        for c in concept_ids_pred:
          f.write(id2wrd[c] + ' ')
        f.write('\n')
      f.write('\n')

def res2confusion_word(res_file, nconcept, plot_cm=True, cmap=plt.cm.Blues):
  with open(res_file, 'r') as f:
    res_list = f.read().strip().split('\n')
  with open('concepts_20.txt', 'r') as f:
    concepts = f.read().strip().split()
  concept2id = {c:i for i, c in enumerate(concepts)}
  cm = np.zeros((nconcept, nconcept))
  
  for res in res_list:
    wrds = res.split(' ')[0]
    correct_wrd = wrds[0].split('_')[0]
    if not correct_wrd in concepts and correct_wrd[:-1] in concepts:
      correct_wrd = correct_wrd[:-1]
    elif correct_wrd[:-2] in concepts:
      correct_wrd = correct_wrd[:-2]
    for wrd in wrds[1:]:
      if wrd in concepts:
        cm[concept2id[correct_wrd], concept2id[wrd]] += 1
  np.savetxt('confusion.txt', cm)    
  if plot_cm:
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix for speech2image retrieval')
    plt.colorbar()
    tick_marks = np.arange(len(concepts))
    plt.xticks(tick_marks, concepts, rotation=45)
    plt.yticks(tick_marks, concepts)
    
    plt.figure()
    plt.show()
    plt.ylabel('True Concepts')
    plt.xlabel('Retrieved Concepts')

def res2accuracy(res_word_level_file, concept_file='../../data/concepts.txt', plot_accuracy=False):
  with open(res_word_level_file, 'r') as f:
    res_list = f.read().strip().split('\n')
  with open(concept_file, 'r') as f:
    concepts = f.read().strip().split()
  concepts.append('no-obj')
  concept2id = {c:i for i, c in enumerate(concepts)}
  concept2cor = {c:0 for c in concepts}
  concept2count = {c:0 for c in concepts}
  concept2fp = {c:0 for c in concepts}
  flag1 = 1
  flag2 = 0
  ntokens = 1
  cur_wrds = []
  for res in res_list:
    if res == '':
      for cur_wrd in cur_wrds: 
        if cur_wrd in cor_wrds:
          concept2cor[cur_wrd] += 1
        else:
          concept2fp[cur_wrd] += 1
      cur_wrds = []
      flag1 = 1
      ntokens += 1
      continue
    if flag1:
      cur_sent = res
      flag1 = 0
      flag2 = 1
      continue
    if flag2:
      flag2 = 0
      cor_wrds = res.split()
      for cor_wrd in cor_wrds:
        concept2count[cor_wrd] += 1
    else:
      wrds = res.split()
      for wrd in wrds:
        if not wrd in cur_wrds: 
          cur_wrds.append(wrd)
   
  print(ntokens)
  with open('concept_accuracy.txt', 'w') as f:
    for c in concepts:    
      if concept2count[c] > 0:
        f.write('Accuracy for {}: {}\n'.format(c, concept2cor[c]/concept2count[c])) 
        f.write('True negative for {}: {}\n'.format(c, (ntokens - concept2count[c] - concept2fp[c])/(ntokens - concept2count[c]))) 


if __name__ == '__main__':
  convert_retrieval_res_sent('../recall_at_10_sent_segmented_sum.txt', '../../data/flickr_sentence_segment/ind2sent_test_cleanup.txt', '../dbid2id_test.txt', '../recall_at_10_sent_segmented_sum_readable.txt')
  retrieval_res_concept_level(data_file='/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment/flickr_sent_fbank_penult_segmented.h5', ret_sents_file='../recall_at_10_sent_segmented_sum_readable.txt', concept_file='../../data/concepts.txt', concept_level_ret_res_file='../recall_at_10_sent_segmented_sum_concept_level.txt')
  #res2accuracy(res_word_level_file='../recall_at_10_sent_segmented_sum_correct_concept_level.txt')
  
  find_correct_retrieval('/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment/flickr_sent_fbank_penult_segmented.h5', '../recall_at_10_sent_segmented_sum.txt', '../dbid2id_test.txt', '../recall_at_10_sent_segmented_sum_correct.txt')
  convert_retrieval_res_sent('../recall_at_10_sent_segmented_sum_correct.txt', '../../data/flickr_sentence_segment/ind2sent_test_cleanup.txt', '../dbid2id_test.txt', '../recall_at_10_sent_segmented_sum_correct_readable.txt')
  retrieval_res_concept_level(data_file='/home/lwang114/spring2018/sp2im_word/data/flickr_sentence_segment/flickr_sent_fbank_penult_segmented.h5', ret_sents_file='../recall_at_10_sent_segmented_sum_correct_readable.txt', concept_file='../../data/concepts.txt', concept_level_ret_res_file='../recall_at_10_sent_segmented_sum_correct_concept_level.txt')
