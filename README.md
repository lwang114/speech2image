# speech2image 2.0
This project implements the speech to image network described in the Harwath and Glass 2016 and 2018 papers:  
~~~~~~
David Harwath, Antonio Torralba and James Glass,
Unsupervised learning of spoken language with visual context,
NIPS 2016

David Harwath, Adria Recasens, Didac Suris, Galen Chuang, Antonio Torralba and James Glass,
Jointly discover visual objects and spoken words from raw sensory inputs
ICCV 2018
~~~~~~
The code is written in tensorflow (0.11.0rc0) with tflearn installed and pytorch (0.2.0). Anaconda 3.0 is the python
environment.

Contents: 
#### ./hg16: 
The two-branch network appeared in the 2016 paper written in tensorflow. The network is trained on the flickr8k_audio caption dataset and vgg penultimate feature vectors in http://isle.illinois.edu/sst/data/vgg_flickr8k.html. Please download the datasets and put them in the data directory before running the program.

1. To run the program, use:  
      python semanticembed.py [number of training data] [number of testing data]

2. To test the network with pretrained weights, use:   
      python scnn_test.py [number of test data] (optional*:[test speech file] [test image file] [number of top indices when computing accuracy])

3. The training curve is in train_accuracy_scnn.txt. The pretrained weights are stored in scnn_pmtrs.npz, and will be automatically updated every time you train the network. 

4. Note that the code may take a bit long to run on CPU, so a GPU is recommended.  

5. 1.1 feature: to retrieve image, first run scnn_test.py, and then use:  
      python image_retrieve.py
      
The project extracts the mel-frequency spectrogram from raw speech using the MFCC code from CMU Sphinx-III project. 

#### ./sp2im_word: 
The two-branch network appeared in the 2016 paper written in tflearn and pytorch; Study the effect of word context in learning the multimodal semantic embedding.

#### ./sp2im_meaning: 
Two-branch network appeared in the 2018 paper based on https://github.com/dharwath/DAVEnet-pytorch. 

This project is currently under development, more features will appear later.
