import nltk
from nltk.stem import WordNetLemmatizer

DEBUG = False
class Preprocessor:
  def __init__(self, api_files, output_file):
    self.api_files = api_files
    self.output_file = output_file
    self.data_info = []

  def extract(self):
    raise NotImplementedError

class COCO_Preprocessor(Preprocessor):
  def __init__(self, api_files, output_file = "mscoco_info.json"):
    super().__init__(api_files, output_file)
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

    self.lemmatizer = WordNetLemmatizer()

  def extract(self):
    tag_prefix = 'N'
    for img_id in coco_api.imgToAnns.keys()[:10]:
      pair_info = {}
      
      captions = speech_api.getImgCaptions(img_id) 
      # Extract word segment with window size of 1, 3 and 5
      for caption in captions:
        pair_info['sp_filename'] = caption.filename
        pair_info['text'] = caption.text
        pair_info['nouns'] = []
        pair_info['context3'] = []
        pair_info['context5'] = []
        capt_id = caption.captionID 
        wrd_aligns = caption.timecode.parse()
        tags = nltk.pos_tag(nltk.tokenize(caption.text))

        for i, (wrd_align, tag) in enumerate(zip(wrd_aligns, tags[:-1])):
          wrd = wrd_align['value']
          if tag[1][0] != tag_prefix:
            continue
          wrd = self.lemmatizer.lemmatize(wrd)
            
          begin = wrd_align['begin']
          end = wrd_align['end']

          context3 = [] 
          context5 = []
          begin3_id = max(i - 1, 0)
          end3_id = min(i + 1, len(wrd_aligns) - 1)
          
          begin5_id = max(i - 2, 0)
          end5_id = min(i + 2, len(wrd_aligns) - 1)

          begin_context3 = wrd_aligns[begin3_id]['begin']
          end_context3 = wrd_aligns[end3_id]['end']

          begin_context5 = wrd_aligns[begin5_id]['begin']
          end_context5 = wrd_aligns[end5_id]['end']
          
          for j in range(begin3_id, end3_id):
            context3.append(wrd_aligns[j]['value']) 
          for j in range(begin5_id, end5_id):
            context5.append(wrd_aligns[j]['value'])
          
          if DEBUG:
            print(wrd, tag)
          pair_info['nouns'].append([wrd, begin, end])
          pair_info['context3'].append((context3, begin_context3, end_context3))
          pair_info['context5'].append((context5, begin_context5, end_context5))

      # Extract image bounding boxes
      im_filename = coco_api.loadImgs(int(im_id))[0]['file_name'] 
      pair_info['im_filename'] = im_filename

      ann_ids = coco_api.getAnnIds(img_id)
      anns = coco_api.loadAnns(ann_ids)
      
      for ann in anns:
        cat = coco_api.loadCats(ann['category_id'])[0]['name']
        x, y, w, h = ann['bbox']
        pair_info['bbox'].append((cat, x, y, w, h))

      self.data_info.append(pair_info)

if __name__ == '__main__':
  preproc = COCO_Preprocessor(["annotations/instances_val2014.json",
                              "val2014/val_2014.sqlite3"])
  preproc.extract()
