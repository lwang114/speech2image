# Adapted from https://github.com/dharwath/DAVEnet-pytorch
import json
import librosa
import librosa.display
import numpy as np
import os
from PIL import Image
from scipy.io import wavfile
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from matplotlib.pyplot import *
import shutil
import random
from nltk.tokenize import word_tokenize

# TODO: Load segments and bounding boxes of the speech
# TODO: Compute the mean and std of the pixel values of the images in MSCOCO 
# TODO: Fix the wavfile reading problem in test code
# TODO: Implement method to use raw length for the text loader
DEBUG = False
def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])

class ImageCaptionDataset(Dataset):
  def __init__(self, dataset_json_file, audio_conf=None, image_conf=None):
    """
    Load all the options for feature extraction
    """
    with open(dataset_json_file, 'r') as f:
      data_json = json.load(f)

    self.data_info = data_json['data']
    self.pixel_mean = np.array(data_json['pixel_mean'])
    self.pixel_std = np.array(data_json['pixel_variance']) ** 1/2
    if not audio_conf:
      self.audio_conf = {}
    else:
      self.audio_conf = audio_conf

    if not image_conf:
      self.image_conf = {}
    else:
      self.image_conf = image_conf

    self.image_base_path = self.image_conf.get('image_base_path', './')
    self.audio_base_path = self.audio_conf.get('audio_base_path', './')
    crop_size = self.image_conf.get('crop_size', 224)
    center_crop = self.image_conf.get('center_crop', False)

    if center_crop:
      self.image_resize_and_crop = transforms.Compose(
        [transforms.Resize(256), transform.CenterCrop(224), transforms.ToTensor()])
    else:
      self.image_resize_and_crop = transforms.Compose(
        [transforms.RandomSizedCrop(crop_size), transforms.ToTensor()]) 
    
    # TODO: modify this part to the mean and std of MSCOCO training set
    RGB_mean = self.image_conf.get('RGB_mean', self.pixel_mean) #[119.4, 113.5, 103.6]) #[0.485, 0.456, 0.406])
    RGB_std = self.image_conf.get('RGB_std', self.pixel_std) #[6309.7 ** 1/2, 6101.6 ** 1/2, 6743.0 ** 1/2])#[0.229, 0.224, 0.225])
    self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    self.windows = {'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett}

  def _LoadAudio(self, path):
    """
    Extract spectrogram feature for speech
    """
    audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
    preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
    sample_rate = self.audio_conf.get('sample_rate', 16000)
    window_size = self.audio_conf.get('window_size', 0.025)
    window_stride = self.audio_conf.get('window_stride', 0.01)
    window_type = self.audio_conf.get('window_type', 'hamming')
    num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
    target_length = self.audio_conf.get('target_length', 1024)
    use_raw_length = self.audio_conf.get('use_raw_length', False)
    padval = self.audio_conf.get('padval', 0)
    fmin = self.audio_conf.get('fmin', 20)
    n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    if DEBUG:
      print(path)
      print(os.path.realpath(path))
    y, sr = librosa.load(path, sample_rate)
    if y.size == 0:
      y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, 
        win_length=win_length, hop_length=hop_length,
        window=self.windows.get(window_type, self.windows['hamming']))    
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
      mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin) 
      melspec = np.dot(mel_basis, spec)
      logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
      logspec = librosa.power_to_db(spec, ref=np.max)
    else:
      raise ValueError('Unknown Audio Feature Type')

    n_frames = logspec.shape[1]
    if use_raw_length:
      target_length = n_frames
    p = target_length - n_frames
    if p > 0:
      logspec = np.pad(logspec, ((0,0), (0,p)), 'constant',
        constant_values=(padval,padval))
    elif p < 0:
      logspec = logspec[:, 0:p]
      n_frames = target_length
    logspec = torch.FloatTensor(logspec)
    return logspec, n_frames

  def _LoadImage(self, impath):
    """
    Resize, crop and normalize image
    """
    img = Image.open(impath).convert('RGB')
    img = self.image_resize_and_crop(img)
    img = self.image_normalize(img)
    return img

  def __getitem__(self, index):
    """
    returns: image, spectrogram, nframes
    """
    datum_info = self.data_info[index]    
    wavpath = datum_info['sp_filename']
    # When one image corresponds to multiple captions, randomly sample one of them 
    if isinstance(wavpath, list):
      wav_ind = random.randint(0, len(wavpath)-1) 
      wavpath = wavpath[wav_ind]
    imgpath = datum_info['im_filename']
    spec, nframes = self._LoadAudio(self.audio_base_path + wavpath)
    image = self._LoadImage(self.image_base_path + imgpath)
    return image, spec, nframes

  def __len__(self):
    return len(self.data_info)

class ImageTextDataset(Dataset):
  def __init__(self, dataset_json_file, text_configs=None, image_configs=None):
    with open(dataset_json_file, 'r') as f:
      data_json = json.load(f)

    self.data_info = data_json['data']

    self.text_base_path = text_configs.get('text_base_path', '.')
    self.target_length = text_configs.get('target_length', 15)
    word_to_idx_json = text_configs.get('word_to_idx_json', None)
    
    if not word_to_idx_json:
      self.word_to_idx = self._Map_Word_To_Idx(self.data_info)
    else:
      with open(self.text_base_path + word_to_idx_json, 'r') as f:
        self.word_to_idx = json.load(f)
    self.n_vocabs = len(self.word_to_idx.keys())
    self.pixel_mean = np.array(data_json['pixel_mean'])
    self.pixel_std = np.array(data_json['pixel_variance']) ** 1/2
    self.image_normalize = transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)

    center_crop = image_configs.get('center_crop', False)
    crop_size = image_configs.get('crop_size', 224)
    self.image_base_path = image_configs.get('image_base_path', '.') 
    if center_crop:
      self.resize_and_crop = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(crop_size),
         transforms.ToTensor()])
    else:
      self.resize_and_crop = transforms.Compose(
        [transforms.RandomSizedCrop(crop_size),
        transforms.ToTensor()])

  def _Map_Word_To_Idx(self, data_info, output_file='word_to_idx.json'):
    vocabs = []
    for item in data_info:
      sent = item['text']
      words = word_tokenize(sent)
      vocabs += words 
    vocabs = set(vocabs)
    n = len(vocabs)
    word_to_idx = {w:i for i, w in enumerate(vocabs)}
    word_to_idx['<UNK>'] = n
    word_to_idx['<PAD>'] = n + 1
    with open(self.text_base_path + output_file, 'w') as f:
      json.dump(word_to_idx, f, indent=4, sort_keys=True)

    return word_to_idx
  
  def _Load_Image(self, impath):
    img = Image.open(self.image_base_path + impath).convert('RGB')
    img = self.resize_and_crop(img)
    img = self.image_normalize(img)
    return img

  def _Load_Text(self, sent):  
    words = word_tokenize(sent)
    indices = [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['<UNK>'] for w in words]
    raw_length = len(indices)
    p = self.target_length - raw_length
    n_words = 0
    if p > 0:
      indices += [self.n_vocabs - 1] * p
      n_words = raw_length
    else:
      indices = indices[:target_length]
      n_words = target_length

    indices = torch.LongTensor(indices)
    if DEBUG:
      print(indices.size())
    return indices, n_words

  def __getitem__(self, index):
    sent = self.data_info[index]['text']
    impath = self.data_info[index]['im_filename']
    indices, n_words = self._Load_Text(sent)
    img = self._Load_Image(impath)
    return img, indices, n_words   

  def __len__(self):
    return len(self.data_info)

if __name__ == '__main__':
  create_dataset = True
  if create_dataset:
    y = np.random.normal(size=(64, 32000))
    img = np.random.uniform(size=(64, 500, 500))
    data_info = {'data':[],
               'pixel_mean': [0., 0., 0.], 
               'pixel_variance': [1., 1., 1.]}
  
    for i in range(64):
      librosa.output.write_wav('%s_%d.wav' % ('../../data/test/test_random', i), y[i], 16000, norm=True)
      img_obj = Image.fromarray(img[i]).convert('RGB')
      img_obj.save('%s_%d.png' % ('../../data/test/test_random', i)) 

      pair_info = {'sp_filename':'%s_%d.wav' % ('test_random', i), 
                   'im_filename':'%s_%d.png' % ('test_random', i),
                   'text': 'hello word %d' % i}           
      data_info['data'].append(pair_info)
    
    #data_info = {'data':[{'sp_filename':'test_random.wav', 'im_filename':'test_random.png'}]*64, 'pixel_mean': [0., 0., 0.], 'pixel_variance': [1., 1., 1.]}
    with open('../../data/test/data_info_train.json', 'w') as f:
      json.dump(data_info, f, indent=4, sort_keys=True)
    shutil.copy('../../data/test/data_info_train.json', '../../data/test/data_info_val.json')

  cwd = os.getcwd()
  audio_config = {'audio_base_path': cwd + '/../../data/test/'}
  image_config = {'image_base_path': cwd + '/../../data/test/'}
  text_config = {'text_base_path': cwd + '/../../data/test/'}
  if not create_dataset:
    text_config['word_to_idx_json'] = 'word_to_idx.json'

  dset = ImageCaptionDataset(cwd + '/../../data/test/data_info_train.json', audio_config, image_config)
  dset2 = ImageTextDataset(cwd + '/../../data/test/data_info_train.json', text_config, image_config)
  
  image, spec, nframes = dset[0]
  print(spec.size())
  
  image, word_indices, n_words = dset2[0]
  print(word_indices)
  #librosa.display.specshow(librosa.power_to_db(spec, ref=np.max),
  #                       y_axis='mel', fmax=8000,
  #                       x_axis='time')
  #colorbar(format='%+2.0f dB') 
