import torch
import argparse
import os
from steps.utils import *
from steps.traintest import *
from models.SpeechEncoders import *
from models.ImageEncoders import *
from dataloaders.image_caption_dataset import *
#from utils.preprocessor import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='../data/test/data_info_train.json',
        help="training data json")
parser.add_argument("--data-val", type=str, default='../data/test/data_info_val.json',
        help="validation data json")
parser.add_argument("--exp-dir", type=str, default="../data/test/exp",
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

args = parser.parse_args()
print(args)
print(args.pretrained_image_model)

# Create a random dataset
cwd = os.getcwd()
audio_conf = {'audio_base_path': cwd + '/../data/test/'}
image_conf = {'image_base_path': cwd + '/../data/test/'}

dset_train = ImageCaptionDataset(cwd + '/../data/test/data_info_train.json', audio_conf, image_conf)
dset_val = ImageCaptionDataset(cwd + '/../data/test/data_info_test.json', audio_conf, image_conf)

train_loader = torch.utils.data.DataLoader(
    dset_train, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    dset_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
) 

audio_model = Davenet()
image_model = VGG16()
if DEBUG:
  print(audio_model.parameters(), image_model.parameters())

train(audio_model, image_model, train_loader, val_loader, args)
#recalls = validate(audio_model, image_model, val_loader, args)
print(recalls)
