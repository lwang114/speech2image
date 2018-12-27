#-*- coding: utf-8 -*-
import json
from pycocotools.coco import COCO
from SpeechCoco.speechcoco_API.speechcoco.speechcoco import SpeechCoco

DEBUG = False
instance_json_path = "annotations/instances_val2014.json"
speech_sql_file = "val2014/val_2014.sqlite3" 

concept_file = "concepts.json"
output_file = "mscoco_wrd_obj_info.json"

#metadata = dict()
#with open(instance_json_path, "r") as f:
#  metadata = json.load(f)
   
with open(concept_file, "r") as f:
  concepts = f.read().strip().split("\n")

concept2cat = dict()
with open(concept_file, "r") as f:
  concept2cat = json.load(f)
#for c_line in concepts:
#  con, cat = c_line.split(' ')
#  concept2cat[con] = cat

# Create a dictionary to map caption-image files to corresponding word 
# boundaries and bounding boxes  
pair2info = dict()

try:
  coco_api = COCO(instance_json_path)
  speech_api = SpeechCoco(speech_sql_file)
  concepts = concept2cat.keys()
except:
  raise RuntimeError("Run make in the pythontools dir of cocoapi before running this")

for img_id in coco_api.imgToAnns.keys():
  captions = speech_api.getImgCaptions(img_id) 
  for caption in captions:
    capt_id = caption.captionID 
    wrd_aligns = caption.timecode.parse()
    for wrd_align in wrd_aligns:
      wrd = wrd_align['value']
      #if DEBUG:
      #  print(wrd)
    
      flag = 0
      if wrd in concept2cat.keys():
        flag = 1
      elif wrd[:-1] in concept2cat.keys() and wrd[-1] == 's':
        flag = 1
        wrd = wrd[:-1]
      elif wrd[:-2] in concept2cat.keys() and wrd[-2:] == 'es':
        flag = 1
        wrd = wrd[:-2]
      elif wrd[:-3]+'y' in concept2cat.keys() and wrd[-3:] == 'ies':
        flag = 1
        wrd = wrd[:-3] + 'y'

      if flag:
        begin = wrd_align['begin']
        end = wrd_align['end']
        if DEBUG:
          print(wrd)
        pair_id = str(capt_id)+'_'+str(img_id)+'_'+wrd

        ann_ids = coco_api.getAnnIds(img_id)
        anns = coco_api.loadAnns(ann_ids)

        for ann in anns:
          if DEBUG:
            print(coco_api.loadCats(ann['category_id']), concept2cat[wrd])
          cat = coco_api.loadCats(ann['category_id'])[0]['name']
          if cat == concept2cat[wrd]:
            if pair_id not in pair2info.keys():
              pair2info[pair_id] = dict()
        
            x, y, w, h = ann['bbox']
            print(pair_id)
            pair2info[pair_id]['timechunk'] = (begin, end)
            pair2info[pair_id]['bbox'] = (x, y, w, h)
            break

with open(output_file, 'w') as f:
  json.dump(pair2info, f)
