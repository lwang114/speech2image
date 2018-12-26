from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
#from scnn_test import *

def scnn_test(X_test, Z_test_vgg, ntop):#(ntx, sp_test, im_test, ntop):
    
    nmf = X_test[0].shape[0]
    nframes = X_test[0].shape[1]
    ntx = X_test.shape[0]
    print('number of data in X_test and Z_test_vgg:', X_test.shape, Z_test_vgg.shape)
    #print('the current frame size and number of mel frequencies are:', nframes, nmf)
    X_test_4d = np.reshape(X_test[0:ntx], [ntx, 1, nframes, nmf])
    
    nlabel = 61
    N = [4, 24, 24]
    J = [nmf, 64, 512, 1024, nlabel]
    D = [nframes, nframes/2, nframes/4, nframes/4]
    npenult_vgg = 4096
    nembed = 1024
    
    w_in = tf.placeholder(tf.float32, shape=[1, N[0]+1, J[0], J[1]])
    b_in = tf.placeholder(tf.float32, shape=[J[1]])
    
    w_hidden1 = tf.placeholder(tf.float32, shape=[1, N[1]+1, J[1], J[2]])
    b_hidden1 = tf.placeholder(tf.float32, shape=[J[2]])
    
    w_hidden2 = tf.placeholder(tf.float32, shape=[1, N[2]+1, J[2], J[3]])
    b_hidden2 = tf.placeholder(tf.float32, shape=[J[3]])
    
    w_out = tf.placeholder(tf.float32, shape=[J[3], J[4]])
    b_out = tf.placeholder(tf.float32, shape=[J[4]])
    
    X_in = tf.placeholder(tf.float32, shape=[None, 1, nframes, nmf])
    X_mean = tf.reduce_mean(X_in)
    X_zm = X_in - X_mean
    
    a1_conv = tf.nn.conv2d(X_zm, w_in, strides=[1, 1, 1, 1], padding='SAME') + b_in
    # Max pooling with vertical stride 1 and horizontal stride 2
    a1_pool = tf.nn.max_pool(a1_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h1 = tf.nn.relu(a1_pool)
    
    a2_conv = tf.nn.conv2d(h1, w_hidden1, strides=[1, 1, 1, 1], padding='SAME') + b_hidden1
    a2_pool = tf.nn.max_pool(a2_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h2 = tf.nn.relu(a2_pool)
    
    a3_conv = tf.nn.conv2d(h2, w_hidden2, strides=[1, 1, 1, 1], padding='SAME') + b_hidden2
    h3 = tf.nn.relu(a3_conv)
    
    # Penultimate layer
    h4 = tf.nn.max_pool(h3, ksize=[1, 1, D[3], 1], strides=[1, 1, 1, 1], padding='VALID')
    h4_re = tf.reshape(h4, [-1, J[3]])
    # L2 normalization
    h4_ren = tf.nn.l2_normalize(h4_re, dim=1)
    
    a_out = tf.matmul(h4_ren, w_out) + b_out
    Y_pred = tf.nn.softmax(a_out)
    
    # Map the penultimate vgg vector to the semantic space
    Z_penult_vgg = tf.placeholder(tf.float32, shape=[None, npenult_vgg])
    w_embed = tf.placeholder(tf.float32, shape=[npenult_vgg, nembed])
    b_embed = tf.placeholder(tf.float32, shape=[nembed])
    Z_embed_vgg = tf.matmul(Z_penult_vgg, w_embed) + b_embed
    
    s_a = tf.matmul(h4_ren, tf.transpose(Z_embed_vgg))
    s = tf.nn.relu(s_a)
    # assume the number of features is nbatch
    s_p = tf.diag_part(s)
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Load network parameters
    data = np.load('scnn_pmtrs.npz')
    pmtrs = data['arr_0']
    _w_in = pmtrs[0]
    _b_in = pmtrs[1]
    _w_hidden1 = pmtrs[2]
    _b_hidden1 = pmtrs[3]
    _w_hidden2 = pmtrs[4]
    _b_hidden2 = pmtrs[5]
    _w_out = pmtrs[6]
    _b_out = pmtrs[7]
    
    data = np.load('vgg_pmtrs.npz')
    pmtrs_vgg = data['arr_0']
    _w_embed = pmtrs_vgg[0]
    _b_embed = pmtrs_vgg[1]
    
    batch_size = 128
    nbatch = int(ntx/batch_size)
    _Z_embed_sp = np.zeros((ntx, nembed))
    _Z_embed_vgg = np.zeros((ntx, nembed))
    #_h3 = np.zeros((ntx, nembed, D[3]))
    for k in range(nbatch):
        X_batch = X_test_4d[batch_size*k:batch_size*(k+1)]
        Z_batch = Z_test_vgg[batch_size*k:batch_size*(k+1)]
        Z_curr_sp = sess.run(h4_ren, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
        Z_curr_vgg = sess.run(Z_embed_vgg, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
        _Z_embed_sp[batch_size*k:batch_size*(k+1)] = Z_curr_sp
        _Z_embed_vgg[batch_size*k:batch_size*(k+1)] = Z_curr_vgg
    
    #_h3_batch = sess.run(h3, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
    #_h3[batch_size*k:batch_size*(k+1)] = _h3_batch.reshape((nbatch, nembed, D[3]))
    X_batch = X_test_4d[batch_size*(nbatch):ntx]
    Z_batch = Z_test_vgg[batch_size*(nbatch):ntx]
    Z_curr_sp = sess.run(h4_ren, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
    Z_curr_vgg = sess.run(Z_embed_vgg, feed_dict={w_in:_w_in, b_in:_b_in, w_hidden1:_w_hidden1, b_hidden1:_b_hidden1, w_hidden2:_w_hidden2, b_hidden2:_b_hidden2, w_out:_w_out, b_out:_b_out, w_embed:_w_embed, b_embed:_b_embed, X_in:X_batch, Z_penult_vgg:Z_batch})
    _Z_embed_sp[batch_size*(nbatch):ntx] = Z_curr_sp
    _Z_embed_vgg[batch_size*(nbatch):ntx] = Z_curr_vgg

    similarity = np.maximum(np.dot(_Z_embed_sp, np.transpose(_Z_embed_vgg)), np.zeros((ntx, ntx)))
    # Similarity for annotation
    similarity_ann = np.maximum(np.dot(_Z_embed_vgg, np.transpose(_Z_embed_sp)), np.zeros((ntx, ntx)))
    
    #X_tx_4d = X_stack_tx.reshape([ntx*(nframes-2*nreduce), 1, nwin, nmf])
    #test_accuracy = sess.run(accuracy, feed_dict={X_in:X_tx_4d, Z_in:Z_tx})
    top_indices = np.zeros((ntop, ntx))
    for k in range(ntop):
        # Find the most similar image feature of the speech feature on the penultimate feature space
        cur_top_idx = np.argmax(similarity, axis=1)
        top_indices[k] = cur_top_idx
        # To leave out the top values that have been determined and the find the top values for the rest of the indices
        for l in range(ntx):
            similarity[l][cur_top_idx[l]] = -1
    #similarity[cur_top_idx] = -1;
    print('Top indices for retrieval is: ', top_indices)
    # Find if the image with the matching index has the highest similarity score
    #dev = abs(np.transpose(np.transpose(top10_indices) - np.linspace(0, ntr-1, ntr)))
    dev = abs(top_indices - np.linspace(0, ntx-1, ntx))
    min_dev = np.amin(dev, axis=0)
    print('Minimum deviation from correct label for retrieval is:', min_dev)
    # Count the number of correct matching by counting the number of 0s in dev
    test_accuracy = np.mean(min_dev == 0)
    print('Test accuracy for retrieval is: ', str(test_accuracy))
    
    top_indices_ann = np.zeros((ntop, ntx))
    for k in range(ntop):
        # Find the most similar image feature of the speech feature on the penultimate feature space
        cur_top_idx_ann = np.argmax(similarity_ann, axis=1)
        top_indices_ann[k] = cur_top_idx_ann
        # To leave out the top values that have been determined and find the top values for the rest of the indices
        for l in range(ntx):
            similarity_ann[l][cur_top_idx_ann[l]] = -1;
    print('Top indices for annotation is: ', top_indices_ann)

    # Find if the image with the matching index has the highest similarity score
    #dev = abs(np.transpose(np.transpose(top10_indices) - np.linspace(0, ntr-1, ntr)))
    dev = abs(top_indices_ann - np.linspace(0, ntx-1, ntx))
    min_dev = np.amin(dev, axis=0)
    print('Minimum deviation from correct label for annotation is: ', min_dev)
    
    # Count the number of correct matching by counting the number of 0s in dev
    test_accuracy = np.mean(min_dev == 0)
    print('Test accuracy for annotation is: ', str(test_accuracy))
    
    '''# Compute the relevancy vector over time
        score_over_time = np.zeros((ntx, D[3]))
        for i in range(ntx):
        score_over_time[i] = np.dot(Z_embed_vgg[i], np.transpose(_h3[i]))
        np.savez('score_over_time.npz', score_over_time)'''
    # Save the top indices of image for each of the speech
    np.savez('top_indices_ret.npz', top_indices)
    # Save the top indices of speech for each of the image
    np.savez('top_indices_ann.npz', top_indices_ann)

# This script retrieves the top n images with highest similarity scores for a given speech
def read_file_list(n):
    # Read a dict from index to filename for both image and caption
    file_info = '../data/flickr_audio/wav2capt.txt'
    files_sp = []
    files_im = []
    with open(file_info, 'r') as f:
        for i in range(n):
            files = f.readline()
            files_part = files.split()
            cur_sp = files_part[0]
            cur_im = files_part[1]
            #print(cur_sp, cur_im)
            files_sp.append(cur_sp)
            files_im.append(cur_im)
    #print(cur_im[125:129])
    return files_sp, files_im

def read_captions(captfiles):
    # Read a dict from image file to its text caption
    file_info = '../data/Flickr8k_text/Flickr8k.token.txt'
    text_capts = {}
    with open(file_info, 'r') as f:
        while len(f.readline()) > 0:
            files = f.readline()
            files_part = files.split()
            nparts = len(files_part)
            cur_sp_parts = files_part[0].split('#')
            cur_sp = cur_sp_parts[0]
            #print(len(cur_sp))
            textcap = ''
            for k in range(nparts-1):
                textcap = textcap+files_part[k+1]+' '
            #print(textcap)
            text_capts[cur_sp] = textcap
    return text_capts

def pil2arr(imfile):
    path = '../data/Flicker8k_Dataset/'
    # load the image and return
    im = Image.open(path+imfile)
    #im_data_seq = im.getdata()
    #print(np.array(im_data_seq).shape)
    #im_data_arr = np.array(im_data_seq, dtype=float).reshape(im.size[0], im.size[1], 3)
    im_data_arr = np.array(im)
    #print(im.size)
    #print(im_data_arr.shape)
    #im.show()
    return im_data_arr

def retrieve(captids):
    # Get the top n indices of image of the current speech. For now, the number of captions and images have to be
    # the same
    #scnn_test(captions, images, n)
    data = np.load('top_indices_ret.npz')
    top_ids = data['arr_0']
    [n, ndata] = top_ids.shape
    ncapt = captids.shape[0]
    files_sp, files_im = read_file_list(ndata)
    text_capts = read_captions(files_im)
    
    if n > ndata:
        print('n>ndata, I cant understand how it is possible. My bad')
        return

    # Find the images for the caption and plot it
    for i in range(ncapt):
        #right = (np.amin(np.abs(top_ids[:, i]-i)) == 0)
        cur_ims = []
        # If the image is correctly retrieved, show it and the rest associated with the queried caption
        #if right:
        #print('Top indices', top_ids[:, captids[i]])
        for j in range(n):
            cur_im_idx = int(top_ids[j, captids[i]])
            cur_name_im = files_im[cur_im_idx]
            cur_name_sp = files_sp[i]
            #print('Line 217 the image', cur_name_im, 'is related to the caption', cur_name_sp)
            cur_im = pil2arr(cur_name_im)
            # Merge the image side-by-side
            cur_ims.append(cur_im)
        #np.concatenate((cur_ims, cur_im), axis=1)
        #else:
        #continue
        # Find the caption name
        right_im_name = files_im[i]
        caption = text_capts[right_im_name]
        # Plot the image
        plt.figure()
        nim = len(cur_ims)
        f,axarr = plt.subplots(1, nim)
        for m in range(nim):
            #print('Line 248 type of the image:', cur_ims[m].dtype)
            axarr[m].imshow(cur_ims[m], aspect=1)
            axarr[m].axis('off')
        plt.title(caption)
        plt.show()
        cur_name_parts = cur_name_im.split('.')
        tmp = cur_name_parts[0]
    #np.savez(tmp+'_top'+str(n)+'.npz', cur_ims)

def annotate(imids):
    # Get the top n indices of image of the current speech. For now, the number of captions and images have to be
    # the same
    #scnn_test(captions, images, n)
    data = np.load('top_indices_ann.npz')
    top_ids = data['arr_0']
    [n, ndata] = top_ids.shape
    nim = imids.shape[0]
    files_sp, files_im = read_file_list(ndata)
    text_capts = read_captions(files_im)
    
    if n > ndata:
        print('n>ndata, I cant understand how it is possible. My bad')
        return
    
    # Find the images for the caption and plot it
    for i in range(nim):
        #right = (np.amin(np.abs(top_ids[:, i]-i)) == 0)
        cur_capts = []
        # If the image is correctly retrieved, show it and the rest associated with the queried caption
        #if right:
        #print('Top indices', top_ids[:, imids[i]])
        for j in range(n):
            cur_im_idx = int(top_ids[j, imids[i]])
            cur_name_im = files_im[cur_im_idx]
            cur_name_sp = files_sp[i]
            #print('Line 217 the image', cur_name_im, 'is related to the caption', cur_name_sp)
            curcapt = text_capts[cur_name_im]
            cur_capts.append(curcapt)
            #cur_im = pil2arr(cur_name_im)
            # Merge the image side-by-side
            #np.concatenate((cur_ims, cur_im), axis=1)
            # Print the captions
            print(curcapt)
        print('\n')
        # Print the current image
        plt.figure()
        right_im_name = files_im[imids[i]]
        cur_im_arr = pil2arr(right_im_name)
        plt.imshow(cur_im_arr)
        plt.axis('off')
        plt.show()

'''
    cur_top_idx = np.argmax(similarity, axis=1)
    top_indices[i] = cur_top_idx
    # To leave out the top values that have been determined and find the top values for the rest of the indices
    similarity[cur_top_idx] = -1
    
    ## Plot the Mel frequency spectrogram of the speech along with the top images of the speech
    # Load data
    file_sp = 'captions.npz'#sys.argv[1]
    data = np.load(file_sp)
    captions = data['arr_0']
    
    file_im = 'images.npz'#sys.argv[2]
    #file_im = 'images.npz'
    data = np.load(file_im)
    images = data['arr_0']
    
    ntop = 5
    #int(sys.argv[3])
    #if len(sys.argv) > 4:
    #    nsel = int(sys.argv[4])
    #    captions = captions[0:nsel]
    #    images = images[0:nsel]
    captions = captions[0:200]
    images = images[0:200]
    
    retrieve(captions, images, ntop)
    
    '''
# Load top indices
#data = np.load('top_indices_im.npz')
#top_indices = np.transpose(data['arr_0'])

# Find the indices corresponding to captions correctly mapped to its image
captions_dat = np.load('captions.npz')
captions = captions_dat['arr_0']

good_ids = []
data = np.load('top_indices_ret.npz')
top_ids = data['arr_0']
ndata = top_ids.shape[1]
for i in range(ndata):
    if not np.amin(np.abs(top_ids[:, i]-i)):
        good_ids.append(i)

good_ids = np.array(good_ids)
retrieve(good_ids)

# Plot the MFCC of the good captions
for l in range(len(good_ids)):
    plt.figure()
    mfcc = captions[int(good_ids[l])]
    plt.imshow(mfcc, cmap=plt.get_cmap('gray'), aspect='auto')


good_ids = []
data = np.load('top_indices_ann.npz')
top_ids = data['arr_0']
ndata = top_ids.shape[1]
for i in range(ndata):
    if not np.amin(np.abs(top_ids[:, i]-i)):
        good_ids.append(i)

good_ids = np.array(good_ids)
annotate(good_ids)
#ncor_capt = good_indices.shape[0]
#nplot = 10
'''for j in range(nplot):
    retrieve(top_indices[:,good_indices[j]])
    cur_capt = captions[good_indices[j]]
    # Plot the mel-freq spectrogram
    plt.figure()
    plt.imshow(cur_capt, intepolation='linear', aspect_ratio='auto')
    '''

'''
    # Image captioning
    data = np.load('top_indices_sp.npz')
    top_indices_sp = np.transpose(data['arr_0'])
    correct_indices = np.linspace(0, ncapt-1, ncapt)
    correct = (np.amin(np.abs(top_indices_sp-)) == 0)
    ncor = correct.shape[0]
    top_correct_indices = []
    
    for i in range(ncor):
    if correct[i] == 1:
    top_is.append(i)
    
    top_correct_indices = np.array(top_correct_indices)
    ncor_capt = .shape[0]
    nplot = 10
    
    for j in range(nplot):
    find_caption(top_indices[:,top_correct_indices[j]])
    cur_im = 
    plt.figure()
    plt.imshow(captions, intepolation='linear', aspect_ratio='auto')'''