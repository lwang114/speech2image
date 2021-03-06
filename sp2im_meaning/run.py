import torch
import argparse
from steps.utils import *
from steps.traintest import *
from models.SpeechEncoders import *
from models.ImageEncoders import *
from models.TextEncoders import *
from dataloaders.image_caption_dataset import *
#from utils.preprocessor import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train-audio", type=str, default='../data/test/',
        help="training directory for audio data (.wav)")
parser.add_argument("--data-train-image", type=str, default='../data/test/',
        help="training directory for image data (.png or .jpg)")
parser.add_argument("--data-val-audio", type=str, default='../data/test/',
        help="validation directory for audio data (.wav)")
parser.add_argument("--data-val-image", type=str, default='../data/test/',
        help="validation directory for image data (.png or .jpg)")

parser.add_argument("--data-info-train", type=str, default='../data/test/data_info_train.json')
parser.add_argument("--data-info-val", type=str, default='../data/test/data_info_val.json')
parser.add_argument("--exp-dir", type=str, default="../data/test",
        help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=100, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
        help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="VGG16",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", type=bool, default=True,
    help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
# Save matchmap
parser.add_argument("--save_matchmap", action="store_true", help="save the matchmaps in the exp_dir if true")
parser.add_argument("--use-text-model", action="store_true", help="use the text-to-image model")
parser.add_argument("--create-word-to-idx", action="store_true", help="create the mapping from word to integer index")
parser.add_argument("--unsampled-loss", action="store_true", help="Use the margin rank loss without sampling")

args = parser.parse_args()
'''args.exp_dir = '../data/mscoco'

args.data_train_audio = '../data/mscoco/val2014/wav/'
args.data_val_audio = '../data/mscoco/val2014/wav/'
args.data_train_image = '../data/mscoco/val2014/imgs/val2014/'
args.data_val_image = '../data/mscoco/val2014/imgs/val2014/'
'''
audio_conf_train = {'audio_base_path': args.data_train_audio}
image_conf_train = {'image_base_path': args.data_train_image}
text_conf_train = {'text_base_path': args.data_train_audio}
if not args.create_word_to_idx:
  text_conf_train['word_to_idx_json'] = 'word_to_idx.json'

audio_conf_val = {'audio_base_path': args.data_val_audio}
image_conf_val = {'image_base_path': args.data_val_image}
text_conf_val = {'text_base_path': args.data_train_audio}
text_conf_val['word_to_idx_json'] = 'word_to_idx.json'  

'''args.data_info_train = '../data/mscoco/train_mscoco_info.json'
args.data_info_val = '../data/mscoco/val_mscoco_info.json'
'''
print(args)

dset_train = None
dset_val = None
if args.use_text_model:
  dset_train = ImageTextDataset(args.data_info_train, text_conf_train, image_conf_train)
  dset_val = ImageTextDataset(args.data_info_val, text_conf_val, image_conf_val)
else:
  dset_train = ImageCaptionDataset(args.data_info_train, audio_conf_train, image_conf_train)
  dset_val = ImageCaptionDataset(args.data_info_val, audio_conf_val, image_conf_val)

train_loader = torch.utils.data.DataLoader(
    dset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    dset_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
) 

if args.use_text_model: 
  text_model = ConvTextEncoder(dset_train.word_to_idx)
  image_model = VGG16()
  train(text_model, image_model, train_loader, val_loader, args)
else:
  audio_model = Davenet()
  image_model = VGG16()
  if DEBUG:
    print(audio_model.parameters(), image_model.parameters())

  train(audio_model, image_model, train_loader, val_loader, args)
#begin = time.time()
#recalls = validate(audio_model, image_model, val_loader, args)
#print(time.time() - begin)
#print(recalls)
