# speech2image 1.0
This project implements the speech to image network described in the Harwath and Glass 2016 paper. The network is trained on the flickr8k_audio caption dataset and vgg penultimate feature vectors in http://isle.illinois.edu/sst/data/vgg_flickr8k.html. The training accuracy of the network for top 10 retrieval images is 35%, and 28% for top 5 images.

To run the training program, use: 
      python semanticembed.py

To run the test program, use: 
      python sp2im_test.py

The training curve is in train_accuracy_scnn.txt. 

The project is currently under development, more features will appear later.
