import numpy as np

sent_list_dir = 'flickr_sentence_segment_807_2018/'
sent_list_file = 'flickr_sp2im_sentences_cleanup.txt'

with open(sent_list_dir + sent_list_file, 'r') as f:
  a = f.read().strip().split('\n')

tr = a[0:2662]
tx = a[2662:3133]
val = a[3133:]

with open(sent_list_dir + 'ind2sent_train_cleanup.txt', 'w') as f:
  f.write('\n'.join(tr))

with open(sent_list_dir + 'ind2sent_test_cleanup.txt', 'w') as f:
  f.write('\n'.join(tx))

with open(sent_list_dir + 'ind2sent_val_cleanup.txt', 'w') as f:
  f.write('\n'.join(val))
