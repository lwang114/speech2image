import json

# Create a dictionary to map image files to caption files to ensure only one image of the same type is in the database
im2sent = dict()
savedir = 'flickr_sentence_segment_807_2018/'
with open('flickr_sentence_segment_807_2018/ind2sent_train_cleanup.txt', 'r') as f:
  sent_ids = f.read().strip().split('\n')
  inds_sorted = sorted(range(len(sent_ids)), key=lambda i: int(sent_ids[i].split('_')[-1])) 
  sent_ids_sorted = [sent_ids[i] for i in inds_sorted]
  for sent in sent_ids_sorted:
    print(sent)
    im_id = '_'.join(sent.split('_')[:-2])
    if im_id not in im2sent.keys():
      im2sent[im_id] = [sent]
    else:
      im2sent[im_id].append(sent)

print(len(im2sent.keys()))
with open('flickr_sentence_segment_807_2018/im2sent_train.json', 'w') as f:
  json.dump(im2sent, f)

im2sent = dict()
with open('flickr_sentence_segment_807_2018/ind2sent_test_cleanup.txt', 'r') as f:
  sent_ids = f.read().strip().split('\n')
  inds_sorted = sorted(range(len(sent_ids)), key=lambda i: int(sent_ids[i].split('_')[-1])) 
  sent_ids_sorted = [sent_ids[i] for i in inds_sorted]
  for sent in sent_ids_sorted:
    print(sent)
    im_id = '_'.join(sent.split('_')[:-2])
    if im_id not in im2sent.keys():
      im2sent[im_id] = [sent]
    else:
      im2sent[im_id].append(sent)

print(len(im2sent.keys()))
with open('flickr_sentence_segment_807_2018/im2sent_test.json', 'w') as f:
  json.dump(im2sent, f)

im2sent = dict()
with open('flickr_sentence_segment_807_2018/ind2sent_val_cleanup.txt', 'r') as f:
  sent_ids = f.read().strip().split('\n')
  inds_sorted = sorted(range(len(sent_ids)), key=lambda i: int(sent_ids[i].split('_')[-1])) 
  sent_ids_sorted = [sent_ids[i] for i in inds_sorted]
  for sent in sent_ids_sorted:
    print(sent)
    im_id = '_'.join(sent.split('_')[:-2])
    if im_id not in im2sent.keys():
      im2sent[im_id] = [sent]
    else:
      im2sent[im_id].append(sent)

print(len(im2sent.keys()))
with open('flickr_sentence_segment_807_2018/im2sent_val.json', 'w') as f:
  json.dump(im2sent, f) 
