import numpy as np
import json

info_txt_file = 'flickr_im_capt_pairs.txt'
im2info = {}

with open(info_txt_file, 'r') as f:
  infos = f.read().strip().split('\n')

for info in infos:
  info_parts = info.split()
  imid = info_parts[1].split('/')[-1].split('.')[0]
  print(imid)
  if not imid in im2info.keys():
    im2info[imid] = [info_parts[2:]]
  else:
    im2info[imid].append(info_parts[2:])

print(im2info[imid][0])
with open('im2info.json', 'w') as f:
  json.dump(im2info, f)
