import numpy as np
import tensorflow as tf
import tflearn
import h5py

def image_encoder():
  net = tflearn.input_data(shape=[None, 4096])
  net = tflearn.conv_2d(net, 1, [1, 25]) #tflearn.fully_connected(net, 512)
  net = tflearn.avg_pool_1d(incoming, kernel_size=[1, 8])
  return net

def image_encoder_pretrain(ncls=70):
  with tf.scope('image_enc'):
    net = tflearn.input_data(shape=[None, 4096], name='in_1')
    net = tflearn.fully_connected(net, ncls, restore=False, name='fc_1')
    net = tflearn.dropout(net, 0.5, name='dropout_1')
    acc = tflearn.metrics.accuracy
    net = tflearn.regression(net, optimizer='adam', metric=acc, loss='softmax_categorical_crossentropy', learning_rate=0.0001)

  return tflearn.DNN(net, tensorboard_verbose=0)

if __name__ == '__main__':
  h5_feat = h5py.File('/home/lwang114/data/flickr/flickr8k_sp2im_feats/flickr_wrd_fbank_penult_70concepts.h5', 'r')
  concepts = list(h5_feat['train'].keys())
  vgg_feat_tr = h5_feat['train/vgg_penult']
  vgg_feat_tr = np.squeeze(vgg_feat_tr)
  y_tr = h5_feat['train/lbl']

  vgg_feat_val = h5_feat['val/vgg_penult']
  vgg_feat_val = np.squeeze(vgg_feat_val)
  y_val = h5_feat['val/lbl']

  ncls = y_tr.shape[1]
  model = image_encoder_pretrain(ncls=ncls)
  netout = tf.get_collection['image_enc/fc_1']
  netin = tf.get_collection['image_enc/in_1']

  y_tr_ = tf.placeholder(tf.float32, shape=(None, ncls))
  auc = tflearn.objectives.roc_auc_score(net_cls, y_tr_) 

  for i in range(10):
    model.fit(vgg_feat_tr, y_tr, batch_size=32, n_epoch=1, validation_set=(vgg_feat_val, y_val), show_metric=True)
    print(sess.run(auc, feed_dict={netin:vgg_feat_tr, y_tr_:y_tr}))

  #vgg_feat_pred = model.predict(vgg_feat_tr)
  #res_err = np.mean(np.linalg.norm(vgg_feat_tr - vgg_feat_pred, ord=2, axis=1) / np.linalg.norm(vgg_feat_tr, ord=2, axis=1), axis=0)
  #print('Residual over total energy square root: ', res_err)
  #model.save('flickr8k_image_encoder_weights')
