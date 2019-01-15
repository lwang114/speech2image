import math
import pickle
import numpy as np
import torch
import time
from torch.autograd import Variable

DEBUG = False
def calc_recalls(image_outputs, audio_outputs, nframes, simtype='MISA'):
  """
  Compute recall@1, 5, 10 given the embeddings
  :returns recalls: a dict containing retrieval and captioning recall@1,
                    recall@5, recall@10 (A_r{1, 5, 10} and I_r{1, 5, 10})
  """
  n = image_outputs.size(0)

  A_r1, I_r1 = AverageMeter(), AverageMeter()
  A_r5, I_r5 = AverageMeter(), AverageMeter()
  A_r10, I_r10 = AverageMeter(), AverageMeter()
  
  begin = time.time()
  S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes) 
  t1 = time.time() - begin

  begin = time.time()
  _, indices_r10_A2I = S.topk(10, 0)

  t2 = time.time() - begin
  _, indices_r10_I2A = S.topk(10, 1)
  indices_r10_A2I = indices_r10_A2I.data
  indices_r10_I2A = indices_r10_I2A.data

  for i in range(n):
    i_find_correct_A2I = -1
    i_find_correct_I2A = -1
    
    for j in range(10):
      if indices_r10_A2I[j, i] == i:  
        i_find_correct_A2I = j
      
      if indices_r10_I2A[i, j] == i:
        i_find_correct_I2A = j
    
    if i_find_correct_A2I >= 0:
      if i_find_correct_A2I == 0:
        A_r1.update(1)
      else:
        A_r1.update(0)

      if i_find_correct_A2I < 5:
        A_r5.update(1)
      else:
        A_r5.update(0)
      A_r10.update(1)
    else:
      A_r1.update(0)
      A_r5.update(0)
      A_r10.update(0)
      
    if i_find_correct_I2A >= 0:
      if i_find_correct_I2A == 0:
        I_r1.update(1)
      else:
        I_r1.update(0)

      if i_find_correct_I2A < 5:
        I_r5.update(1)
      else:
        I_r5.update(0)
      
      I_r10.update(1)
    else:
      I_r1.update(0)
      I_r5.update(0)
      I_r10.update(0)
  recalls = {'A_r1':A_r1.avg, 'I_r1':I_r1.avg,
             'A_r5':A_r5.avg, 'I_r5':I_r5.avg,
             'A_r10':A_r10.avg, 'I_r10':I_r10.avg}
  return recalls    
  
def computeMatchmap(I, A):
  assert(I.dim() == 3)
  assert(A.dim() == 2) 
  D, H, W = I.size(0), I.size(1), I.size(2)
  T = A.size(1)
  I_flat = I.view(D, -1).t()
  M_flat = torch.mm(I_flat, A)
  M = M_flat.view(H, W, T)
  return M

def matchmapSim(M, simtype):
  """
  Compute the SISA, MISA and SIMA similarity given the match map
  """
  assert(M.dim() == 3)
  S = None
  if simtype == 'SISA':
    S = M.mean()
  elif simtype == 'MISA':
    M_WT, _ = torch.max(M, dim=0)
    M_T, _ = torch.max(M_WT, dim=0)
    S = M_T.mean() 
  elif simtype == 'SIMA':
    M_HW, _ = torch.max(M, dim=2) 
    S = M_HW.mean()
  else:
    raise ValueError('Similarity type not known')

  return S

def sampled_margin_rank_loss(image_output, audio_output, nframes, margin=1., simtype='MISA'):
  """
  Compute the triplet margin ranking loss for each image/caption pair
  The impostor image/caption is randomly sampled from the minibatch
  """
  assert(image_output.dim() == 4)
  assert(audio_output.dim() == 3)

  n = image_output.size(0)
  #loss = torch.zeros(1, device=image_output.device, requires_grad=True)
  loss = Variable(torch.zeros(1), requires_grad=True)
  #margin = Variable(torch.FloatTensor(margin))
  #if DEBUG:
  #  print('margin requires_grad', margin.requires_grad)

  if torch.cuda.is_available():
    loss = loss.cuda()
    image_output = image_output.cuda()
    audio_output = audio_output.cuda()

  for i in range(n):
    # Sampled the imposters 
    i_imp_A = i
    i_imp_I = i
    while (i == i_imp_A):
      i_imp_A = np.random.randint(0, n)
    
    while (i == i_imp_I):
      i_imp_I = np.random.randint(0, n)

    nF = nframes[i]
    nFimp = nframes[i_imp_A]
    A, I = audio_output[i][:, :nF], image_output[i]
    A_imp = audio_output[i_imp_A][:, :nFimp]
    I_imp = image_output[i_imp_I]
     
    S_anchor = matchmapSim(computeMatchmap(I, A), simtype=simtype)
    S_imp_img = matchmapSim(computeMatchmap(I_imp, A), simtype=simtype)
    S_imp_aud = matchmapSim(computeMatchmap(I, A_imp), simtype=simtype)
    if DEBUG:
      print('S_imp_img, S_imp_aud, S_anchor: ', S_imp_img.data, S_imp_aud.data, S_anchor.data)
    
    A2I_simdif = margin + S_imp_img - S_anchor
    if (A2I_simdif.data > 0).all():
      loss = loss + A2I_simdif
    
    I2A_simdif = margin + S_imp_aud - S_anchor
    if (I2A_simdif.data > 0).all():
      loss = loss + I2A_simdif

  loss = loss / n
  
  return loss
  
def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
  """
  Compute the similarity matrix for a batch of image and audio
  """
  assert(image_outputs.dim() == 4)
  assert(audio_outputs.dim() == 3)
  n = image_outputs.size(0)
  #S = torch.zeros(n, n, device=image_outputs.device, requires_grad=False)
  S = Variable(torch.zeros(n, n))
  if torch.cuda.is_available():
    S = S.cuda()
  for image_index in range(n):
    for audio_index in range(n):
      nF = nframes[audio_index]
      S[image_index, audio_index] = matchmapSim(computeMatchmap(image_outputs[image_index], audio_outputs[audio_index][:, 0:nF]), simtype=simtype)

  return S

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
  lr = base_lr * (0.1 ** (epoch // lr_decay))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

if __name__ == '__main__':
  a_out = Variable(torch.randn(20, 1024, 128).cuda(), requires_grad=True)
  i_out = Variable(torch.randn(20, 1024, 14, 14).cuda(), requires_grad=True)
  nframes = [128]*20
  print(calc_recalls(i_out, a_out, nframes, simtype='MISA'))
  print(calc_recalls(i_out, a_out, nframes, simtype='SISA'))
  print(calc_recalls(i_out, a_out, nframes, simtype='SIMA'))
  print(calc_recalls(a_out.unsqueeze(2), a_out, nframes, simtype='MISA'))
  loss = sampled_margin_rank_loss(i_out, a_out, nframes)
  print(loss.data.type())
  print(loss, loss.requires_grad) 
  
  # Test if the gradients are nonzero
  loss.backward()
  print(a_out.grad)
  print(a_out.data.type(), i_out.data.type())
  print(a_out.requires_grad)
  print(a_out.grad[0, 0, :10])
  print(i_out.grad[0, 0, :10, :10])
