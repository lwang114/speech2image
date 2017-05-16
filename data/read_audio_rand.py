import numpy as np
import matplotlib.pyplot as plt
import sys

def read_captions():
    # Read a dict from image file to its text caption
    file_info = '../data/Flickr8k_text/Flickr8k.token.txt'
    text_capts = {}
    with open(file_info, 'r') as f:
        flag = 1
        while flag:
            files = f.readline()
            if not files:
                break
            # Break the info for the current text caption
            files_part = files.split()
            nparts = len(files_part)
            cur_sp_parts = files_part[0].split('#')
            # The first fragment is the filename
            cur_sp = cur_sp_parts[0]
            #print(len(cur_sp))
            textcap = ''
            # Glue the fragments of the sentence together
            for k in range(nparts-1):
                textcap = textcap+files_part[k+1]+' '
            #print(textcap)
            text_capts[cur_sp] = textcap
    return text_capts


wavpath = 'flickr_audio/wav2capt.txt'
qsp = []
qim = []
# Read a dictionary to map image file to text captions
dict_im2tx = read_captions()

# Read the list of image and audio files
with open(wavpath, 'r') as f:
    flag = 1
    while flag:
        line = f.readline()
        if not line:
            break
        line_parts = line.rstrip().split()
        cur_sp = line_parts[0]
        cur_im = line_parts[1]
        qsp.append(cur_sp)
        qim.append(cur_im)

print('Total number of audio: ', len(qsp))
print('Total number of text: ', len(dict_im2tx.keys()))

nid = len(qsp)
nrand = 0
if int(sys.argv[1]) == 1:
    # randomize the indices and pick nrand of them to show
    nrand = int(sys.argv[2])
    ids = np.argsort(np.random.normal(size=[nid,]))
    isel = ids[0:nrand]
else:
    if int(sys.argv[1]) == 0:
        nrand = int(sys.argv[3])-int(sys.argv[2])+1
        ids = np.arange(nid)
        isel = ids[int(sys.argv[2]):int(sys.argv[3])+1]


print('Indices selected: ', isel)

qim_sel = [qim[j] for j in isel]

show_im = 0
impath = 'Flicker8k_Dataset/'

# show the texts and images in the selective list
if show_im:
    for i in range(nrand):
        im = Image.open(impath+qim_sel[i])
        im_arr = np.array(im)
        plt.figure()
        plt.imshow(im_arr)

dict_imtype = dict()
for i in range(nrand):
    # count the number of the different types of images for the current selected list of images
    if qim_sel[i] not in dict_imtype:
        dict_imtype[qim_sel[i]] = 1
    else:
        dict_imtype[qim_sel[i]] = dict_imtype[qim_sel[i]] + 1
    #print(dict_im2tx[qim_sel[i]])
    #print('\n')

print('Number of image types:', len(dict_imtype))
