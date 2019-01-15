# Adapted from https://github.com/dharwath/DAVEnet-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as imagemodels
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from PIL import Image
import numpy as np

DEBUG = False
class Resnet18(imagemodels.ResNet):
  def __init__(self, embedding_dim=1024, pretrained=False):
    super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
      self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))    
    self.avgpool = None
    self.fc = None
    self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
    self.embedding_dim = embedding_dim
    self.pretrained = pretrained

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.embedder(x)
    return x

class Resnet34(imagemodels.ResNet):
  def __init__(self, embedding_dim=1024, pretrained=False):
    super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
    if pretrained:
      self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
    self.pretrained = pretrained
    self.embedding_dim = embedding_dim
    self.avgpool = None
    self.fc = None
    self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
    

  def forward(self, x):  
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.embedder(x)
    return x 

class VGG16(nn.Module):
  def __init__(self, embedding_dim=1024, pretrained=False, method1=True):
    super(VGG16, self).__init__()
    model = None
    if method1:
      model = imagemodels.VGG(imagemodels.vgg.make_layers(imagemodels.vgg.cfg['D']))
      if pretrained:
        model.load_state_dict(model_zoo.load_url(imagemodels.vgg.model_urls['vgg16']))
    else:
      model = imagemodels.__dict__['vgg16'](pretrained=pretrained)

    model = model.features
    model = nn.Sequential(*list(model.children())[:-1])
    last_layer_index = len(list(model.children()))
    model.add_module(str(last_layer_index), 
        nn.Conv2d(512, embedding_dim, kernel_size=3, stride=1, padding=1))
    self.image_model = model

  def forward(self, x):  
    x = self.image_model(x)
    # Additional normalization layer
    x_norm = torch.norm(x, 2, 1, keepdim=True)
    x = x / x_norm
    if DEBUG:
      print(x_norm.size())
    return x

if __name__ == '__main__':
  create_image = False
  img = None
  if create_image:
    x = np.random.uniform(size=(500, 500))
    print(x.dtype)   
    img = Image.fromarray(x).convert('RGB')
    img.save('../../data/test/random2.png')
  else:
    img = Image.open('../../data/test/random2.png')

  
  resize = transforms.Resize(224)
  trans = transforms.Compose([resize, 
                    transforms.ToTensor()])
  img = trans(img)
  # Test Resnet 18
  res18 = Resnet18()
  out = res18.forward(img.unsqueeze(0))
  print(out.shape)
  res18 = Resnet18(pretrained=True)
  out = res18.forward(img.unsqueeze(0))
  print(out.shape)
   
  # Test Resnet 34
  res34 = Resnet34()
  out = res34.forward(img.unsqueeze(0))
  print(out.shape)
  #res34 = Resnet34(pretrained=True)
  #out = res34.forward(img.unsqueeze(0))
  #print(out.shape)

  # Test VGG 16
  vgg16 = VGG16(pretrained=False)
  out = vgg16.forward(img.unsqueeze(0))
  print(out.shape)
  vgg16 = VGG16(pretrained=True)
  out2 = vgg16.forward(img.unsqueeze(0))
  print(out.shape)
  # Test alternate method for using the pretrained net
  vgg16 = VGG16(pretrained=True, method1=False)
  out3 = vgg16.forward(img.unsqueeze(0))
  print(out2.shape)
  print(np.array_equal(out.data.numpy(), out3.data.numpy()))
  print(np.array_equal(out2.data.numpy(), out3.data.numpy()))
