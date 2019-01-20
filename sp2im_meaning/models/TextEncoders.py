# Adapted from https://github.com/dharwath/DAVEnet-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as imagemodels
import torchvision.transforms as transforms
import librosa
import numpy as np

DEBUG = False
# TODO: allow context in the embedding
class ConvTextEncoder(nn.Module):
  def __init__(self, word_to_idx, embedding_dim=1024):
    super(ConvTextEncoder, self).__init__()
    self.word_to_idx = word_to_idx
    self.n_vocabs = len(self.word_to_idx.items())
    self.embed = nn.Embedding(self.n_vocabs, 200)
    self.conv1 = nn.Conv2d(1, 512, kernel_size=(200, 3), stride=(1, 1), padding=(0, 1))
    self.conv2 = nn.Conv2d(512, embedding_dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
  
  def forward(self, word_indices):
    #word_indices = [[self.word_to_idx[w] for w in sent] for sent in inputs]
    #word_indices = Variable(torch.LongTensor(word_indices))
    
    x = torch.transpose(self.embed(word_indices), 1, 2).unsqueeze(1)
    if DEBUG:
      print(x.data.size())
    x = F.relu(self.conv1(x))
    x = self.conv2(x)
    x = x.squeeze(2)
    return x

if __name__ == '__main__':
  create_random_inds = False
  word_to_idx = {'hello':0, 'world':1}
  x = [['hello', 'world']]
  word_indices = [[word_to_idx[w] for w in s] for s in x] 
  word_indices = Variable(torch.LongTensor(word_indices))
  #if create_random_inds:
  #  x = np.random.randint(0, 10000, size=(64, 10000, 10))
  #  np.save('../../data/test/random.npy', x)
  #else:
  #  x = np.load('../../data/test/random.npy')
  #x = x.tolist()
  enc = ConvTextEncoder(word_to_idx)
  out = enc(word_indices)
  print(out.size())
