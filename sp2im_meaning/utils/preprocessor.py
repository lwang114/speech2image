import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
from collections import defaultdict
from PIL import Image
from pycocotools.coco import COCO
from SpeechCoco.speechcoco_API.speechcoco.speechcoco import SpeechCoco

DEBUG = True
class Preprocessor(object):
  def __init__(self, api_files, data_dir, output_file):
    self.api_files = api_files
    self.data_dir = data_dir
    self.output_file = output_file
    self.data_info = []
    self.pixel_mean = 0.
    self.pixel_variance = 0.

  def extract(self):
    raise NotImplementedError

class COCO_Preprocessor(Preprocessor):
  def __init__(self, api_files, data_dir, output_file = "mscoco_info.json", split_ratio=1.):
    super(COCO_Preprocessor, self).__init__(api_files, data_dir, output_file)
    instance_json_path = self.api_files[0]     
    speech_sql_file = self.api_files[1]    
    #concept_file = "concept_pairs.json"
    #concept2cat = dict()
    #with open(concept_file, "r") as f:
    #  self.concept2cat = json.load(f)
    
    # Create a dictionary to map caption-image files to corresponding word 
    # boundaries and bounding boxes  
    try:
      self.coco_api = COCO(instance_json_path)
      self.speech_api = SpeechCoco(speech_sql_file)
    except:
      raise RuntimeError("Run make in the pythontools dir of cocoapi before running this")

    self.ntr = int(split_ratio * len(self.coco_api.imgToAnns.keys()))
    self.lemmatizer = WordNetLemmatizer()

  def calc_pixel_mean_and_variance(self):
    pixel_mean = 0.
    pixel_var = 0.
    n = 0. 
    for img_id in self.coco_api.imgToAnns.keys()[:self.ntr]:
      im_filename = self.coco_api.loadImgs(int(img_id))[0]['file_name']
      if DEBUG:
        print(im_filename)
      img = Image.open("%s/%s" % (self.data_dir, im_filename), 'r').convert('RGB')
      w = np.array(img).shape[0]
      h = np.array(img).shape[1]
      pixel_mean += np.sum(np.sum(np.array(img), axis=0), axis=0)
      n += w * h
    pixel_mean = pixel_mean / n

    for img_id in self.coco_api.imgToAnns.keys():
      im_filename = self.coco_api.loadImgs(int(img_id))[0]['file_name']
      if DEBUG:
        print(im_filename)
      img = Image.open("%s/%s" % (self.data_dir, im_filename), 'r').convert('RGB')
      w = np.array(img).shape[0]
      h = np.array(img).shape[1]
      
      pixel_var += np.sum(np.sum((np.array(img) - pixel_mean) ** 2, axis=0), axis=0)   
    pixel_var = pixel_var / n
    return list(pixel_mean), list(pixel_var)

  def extract(self, split_ratio=1.):
    tag_prefix = 'N'
    puncts = [',', ';', '-', '\"', '\'']
    silence = '__SIL__'
    for img_id in self.coco_api.imgToAnns.keys()[0]:
      pair_info['sp_filename'] = [] 
      pair_info['text'] = [] 
      pair_info['nouns'] = [] 
      pair_info['context3'] = [] 
      pair_info['context5'] = [] 

      captions = self.speech_api.getImgCaptions(img_id) 
      # Extract word segment with window size of 1, 3 and 5
      for k, caption in enumerate(captions):
        pair_info['sp_filename'].append(caption.filename)
        pair_info['text'].append(caption.text)
        pair_info['nouns'].append([])
        pair_info['context3'].append([])
        pair_info['context5'].append([])
        if DEBUG:
          print(caption.filename)
        capt_id = caption.captionID 
        timecode = caption.timecode.parse()
        disfluency = caption.disfluencyVal
        text = nltk.tokenize.word_tokenize(caption.text) 
        text_fluent = [w for w in text if w not in disfluency and w not in puncts]
        timecode_fluent = [wa for wa in timecode if wa['value'] not in disfluency and wa['value'] != silence]

        tags = nltk.pos_tag(text_fluent)
        for i, (wrd_align, tag) in enumerate(zip(timecode_fluent, tags[:-1])):
          wrd = wrd_align['value']
          
          if tag[1][0] != tag_prefix:
            continue
          wrd = self.lemmatizer.lemmatize(wrd)
            
          begin = wrd_align['begin']
          end = wrd_align['end']

          context3 = [] 
          context5 = []
          begin3_id = max(i - 1, 0)
          end3_id = min(i + 1, len(timecode_fluent) - 1)
          
          begin5_id = max(i - 2, 0)
          end5_id = min(i + 2, len(timecode_fluent) - 1)

          begin_context3 = timecode_fluent[begin3_id]['begin']
          end_context3 = timecode_fluent[end3_id]['end']

          begin_context5 = timecode_fluent[begin5_id]['begin']
          end_context5 = timecode_fluent[end5_id]['end']
          
          for j in range(begin3_id, end3_id):
            context3.append(timecode_fluent[j]['value']) 
          for j in range(begin5_id, end5_id):
            context5.append(timecode_fluent[j]['value'])
          
          if DEBUG:
            print(wrd, tag)
          pair_info['nouns'][k].append([wrd, begin, end])
          pair_info['context3'][k].append((context3, begin_context3, end_context3))
          pair_info['context5'][k].append((context5, begin_context5, end_context5))

      # Extract image bounding boxes
      im_filename = self.coco_api.loadImgs(int(img_id))[0]['file_name'] 
      pair_info['im_filename'] = im_filename
      pair_info['bboxes'] = []

      ann_ids = self.coco_api.getAnnIds(img_id)
      anns = self.coco_api.loadAnns(ann_ids)
      
      for ann in anns:
        cat = self.coco_api.loadCats(ann['category_id'])[0]['name']
        x, y, w, h = ann['bbox']
        pair_info['bboxes'].append((cat, x, y, w, h))

      self.data_info.append(pair_info)
    self.pixel_mean, self.pixel_variance = self.calc_pixel_mean_and_variance()
      
    # TODO: Implement cross validation
    n_examples = len(self.data_info)
    if n_examples != self.ntr:
      data_info_train = self.data_info[:self.ntr] 
      data_info_val = self.data_info[self.ntr:]
      with open('train_%s' % self.output_file, 'w') as f:
        json.dump({'data': data_info_train, 
                 'pixel_mean': self.pixel_mean,
                 'pixel_variance': self.pixel_variance}, 
                f, indent=4, sort_keys=True)
      
      with open('val_%s' % self.output_file, 'w') as f:
        json.dump({'data': data_info_val, 
                 'pixel_mean': self.pixel_mean,
                 'pixel_variance': self.pixel_variance}, 
                f, indent=4, sort_keys=True)
    else:
      with open(self.output_file, 'w') as f:
        json.dump({'data': self.data_info, 
                 'pixel_mean': self.pixel_mean,
                 'pixel_variance': self.pixel_variance}, 
                f, indent=4, sort_keys=True)

if __name__ == '__main__':
  preproc = COCO_Preprocessor(["annotations/instances_train2014.json",
                              "../../data/mscoco/train2014/train_2014.sqlite3"],
                              "../../data/mscoco/train2014/imgs/train2014", 
                              output_file='mscoco_info.json')
  '''preproc = COCO_Preprocessor(["annotations/instances_val2014.json",
                              "../../data/mscoco/val2014/val_2014.sqlite3"],
                              "../../data/mscoco/val2014/imgs/val2014", 
                              output_file='mscoco_info.json',
                              split_ratio=0.8)
  '''
  print(preproc.extract())
  #print(preproc.calc_pixel_mean_and_variance())
