from PIL import Image

# This script retrieves the top n images with highest similarity scores for a given speech
def get_file_list(n):
    file_info = '../data/flickr_audio/wav2capt.txt'
    files_sp = []
    files_im = []
    with open(file_info, 'r') as f:
        for i in range(10):
            files = f.readline()
            files_part = files.split()
            file_sp = files_part[0]
            file_im = files_part[1]
            print(files_sp, files_im)
            files_sp.append(files_sp)
            files_im.append(files_im)
    return files_sp, files_im

def get_image(imfile):
    path = '../data/flickr8k/'
    # load the image and return
    im = Image.open(path+imfile)
    im_data_seq = im.showdata()
    im_data_arr = np.array(list(im_data_seq)).reshape(im_size[0], im_size[1])
    im.show()
    return im_data_arr

def find_image(top_indices, files_im):
    # Get the top n indices of image of the current speech
    [ndata, ntop] = top_indices.shape
    if n > ndata:
        return
    # Find the image and plot it
    for i in range(ndata):
        cur_ims = []
        for j in range(ntop):
            cur_name = files_im[j]
            # Load image data (need refine)
            cur_im = get_image(cur_name)
            if j == 0:
                # Add the image to the object?
                cur_ims = [cur_im]
            else:
                # Merge the image side-by-side
                np.concatenate(cur_ims, cur_im, axis=1)
        plt.figure()
        # Plot the image
        plt.imshow(cur_ims)
        cur_name_parts = cur_name.split('.')
        tmp = cur_name_parts[0]
        np.savez(tmp+'_top'+str(ntop)+'.npz', cur_ims)
        '''# Find the most similar image feature of the speech feature on the penultimate feature space
        cur_top_idx = np.argmax(similarity, axis=1)
        top_indices[i] = cur_top_idx
        # To leave out the top values that have been determined and find the top values for the rest of the indices
        similarity[cur_top_idx] = -1

        # Based on the indices, find the file associated with the indices, assuming the indices is taken from capt2txt file in the regular order
        # Store the image data in an array'''

## Plot the Mel frequency spectrogram of the speech along with the top images of the speech
# Load data
data = np.load('captions.npz')
captions = data['arr_0']
ncapt = captions.shape[0]

# Load top indices
data = np.load('top_indices_im.npz')
top_indices = np.transpose(data['arr_0'])

# Find the indices corresponding to captions correctly mapped to its image
correct_indices = np.linspace(0, ncapt-1, ncapt)
correct = (np.amin(np.abs(top_indices-correct_indices)) == 0)
ncor = correct.shape[0]
good_indices = []
for i in range(ncor):
    if correct[i] == 1:
        good_indices.append(i)

good_indices = np.array(good_indices)
ncor_capt = good_indices.shape[0]
nplot = 10
for j in range(nplot):
    find_image(top_indices[:,good_indices[j]])
    cur_capt = captions[good_indices[j]]
    # Plot the mel-freq spectrogram
    plt.figure()
    plt.imshow(cur_capt, intepolation='linear', aspect_ratio='auto')

'''# Image captioning
data = np.load('top_indices_sp.npz')
top_indices_sp = np.transpose(data['arr_0'])
correct_indices = np.linspace(0, ncapt-1, ncapt)
correct = (np.amin(np.abs(top_indices_sp-correct_indices)) == 0)
ncor = correct.shape[0]
top_correct_indices = []

for i in range(ncor):
    if correct[i] == 1:
        top_correct_indices.append(i)

top_correct_indices = np.array(top_correct_indices)
ncor_capt = top_correct_indices.shape[0]
nplot = 10

for j in range(nplot):
    find_caption(top_indices[:,top_correct_indices[j]])
    cur_im = 
    plt.figure()
    plt.imshow(captions, intepolation='linear', aspect_ratio='auto')'''
