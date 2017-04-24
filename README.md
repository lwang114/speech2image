# speech2image 1.0
This project implements the speech to image network described in the Harwath and Glass 2016 paper. The network is trained on the flickr8k_audio caption dataset and vgg penultimate feature vectors in http://isle.illinois.edu/sst/data/vgg_flickr8k.html. The training accuracy of the network for top 10 retrieval images is 35%, and 28% for top 5 images.

1. To run the program, use: 
      python semanticembed.py

2. The audio caption dataset is quite large, about 8GB. If you do not want to download it with the network code, just download the SemEmbedding folder, and create a new sp2im folder yourselves

3. The training curve is in train_accuracy_scnn.txt. The pretrained weights are stored in scnn_pmtrs.npz, and will be automatically updated every time you train the network. 

The project extracts the mel-frequency spectrogram from raw speech using the MFCC code from CMU Sphinx-III project. This project is currently under development, more features will appear later.
