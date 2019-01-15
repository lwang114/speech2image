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
class Davenet(nn.Module):
  def __init__(self, embedding_dim=1024, normalize=True):
    super(Davenet, self).__init__()
    self.embedding_dim = embedding_dim
    self.normalize = normalize 
    self.bn = nn.BatchNorm2d(1)
    self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
    self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0))
    self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5))
    self.conv3 = nn.Conv2d(256, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
    self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
    self.embed = nn.Conv2d(512, embedding_dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))

  def forward(self, x):
    if x.dim() == 3:
      x = x.unsqueeze(1)
    x = self.bn(x)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.pool(x)
    x = F.relu(self.conv4(x))
    x = self.pool(x)
    x = F.relu(self.embed(x))
    x = self.pool(x)
    x = x.squeeze(2)
    # Additional normalization layer
    x_norm = torch.norm(x, 2, 1, keepdim=True)
    if self.normalize:
      x = x / x_norm
    return x

if __name__ == '__main__':
  '''create_audio = False
  y = None
  sr = 16000
  if create_audio:
    y = np.random.normal(size=(16000,))
    librosa.output.write_wav('../../data/test/random2.wav', y, sr)
  else:
    y, sr = librosa.core.load('../../data/test/random2.wav', sr=sr)
  x = librosa.feature.mfcc(y, sr, n_mfcc=40)
  padlen = 1024 - x.shape[1]
  if padlen > 0:
    x = np.pad(x, ((0, 0), (0, padlen)), mode='constant', constant_values=0)
  else:
    x = x[:, :1024]
  '''
  x = np.random.normal(size=(20, 40, 1024))
  x = torch.FloatTensor(x).unsqueeze(0)

  davenet = Davenet()
  out = davenet.forward(x)
  print(out.shape)
