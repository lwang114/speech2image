import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
from .utils import *

DEBUG = True
def train(audio_model, image_model, train_loader, test_loader, args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.set_grad_enabled(True)
  
  batch_time = AverageMeter()
  data_time = AverageMeter() 
  loss_meter = AverageMeter()
  progress = []
  best_epoch, best_acc = 0, -np.inf
  global_step, epoch = 0, 0
  start_time = time.time()
  exp_dir = args.exp_dir

  def _save_progress():
    progress.append([epoch, global_step, best_epoch, best_acc,
            time.time() - start_time])
    with open("%s/progress.pkl" % exp_dir, "wb") as f:
      pickle.dump(progress, f)

  # Load the current progress, epoch, global step, best epoch and 
  # best acc
  if args.resume:
    progress_pkl = "%s/progress.pkl" % exp_dir
    progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl) 
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)
  
  # Parallelize the models
  if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)
  
  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)

  # Load the current weights of the models
  if epoch != 0:
    audio_model.load_state_dict("%s/models/audio_model.%d.pth" % (exp_dir, epoch))
    image_model.load_state_dict("%s/models/image_model.%d.pth" % (exp_dir, epoch))
    
  # Assign devices for the models (.to())
  audio_model.to(device)
  image_model.to(device)

  # Find the parameters to be optimized (decide based on .requires_grad) 
  audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
  image_trainables = [p for p in image_model.parameters() if p.requires_grad]
  trainables = audio_trainables + image_trainables
  if DEBUG:
    print(trainables)
  # Set up the optimizer (SGD / ADAM);
  optimizer = None
  if args.optim == 'sgd':
    optimizer = torch.optim.SGD(trainables, 
                lr = args.lr, 
                momentum = args.momentum,
                weight_decay = args.weight_decay)
  elif args.optim == 'adam':
    optimizer = torch.optim.Adam(trainables,
                lr = args.lr,
                betas = (0.95, 0.999),
                weight_decay = args.weight_decay) 
  else:
    raise ValueError('Optimizer %s is not supported' % args.optim)
  
  # States are stored in the double dictionary [optimizer].state; assign a device 
  # for each state variable 
  # Load the current states of the optimizer in /models/optim_state.[epoch].pth
  if epoch != 0:
    optimizer.load_state_dict('%s/models/optim_state.%d.pth' % (args.exp_dir, best_epoch))
    for s in optimizer.state.values():
      for k, v in s.items():
        if isinstance(v, torch.Tensor):
          s[k] = v.to(device)
          print("loaded state dict from epoch %d" % epoch)

  epoch += 1
    
  print("current #steps=%s, #epochs=%s" % (global_step, epoch))
  print("start training...")

  audio_model.train()
  image_model.train()
  while epoch < args.n_epochs:
    # Adjust learning rate
    adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
    end_time = time.time()
    audio_model.train()
    image_model.train()
    # Batch training
    for i, (image_inputs, audio_inputs, nframes) in enumerate(train_loader):  
      data_time.update(time.time() - end_time)
      image_inputs = image_inputs.to(device)
      audio_inputs = audio_inputs.to(device)
      B = image_inputs.size(0) 

      optimizer.zero_grad()

      image_outputs = image_model(image_inputs)
      audio_outputs = audio_model(audio_inputs)

      # Sampled margin loss
      pooling_ratio = round(audio_inputs.size(-1) / audio_outputs.size(-1))
      nframes.div_(pooling_ratio)
      loss = sampled_margin_rank_loss(
                image_outputs, audio_outputs, nframes,
                margin = args.margin, simtype = args.simtype) 
      
      loss.backward()
      optimizer.step()

      # Record loss
      loss_meter.update(loss.item(), B)
      batch_time.update(time.time() - end_time)

      # Print the result every n_print_steps
      if global_step != 0 and global_step % args.n_print_steps == 0:
        sys.stdout.flush()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss_meter=loss_meter))
        if np.isnan(loss_meter.avg):
          print("training diverged...")
          return
      
      end_time = time.time()
      global_step += 1
      
    # Validation
    recalls = validate(audio_model, image_model, test_loader, args)
    # Save models and optimizers
    torch.save(audio_model.state_dict(), "%s/models/audio_model.%d" % (exp_dir, epoch))
    torch.save(image_model.state_dict(), "%s/models/image_model.%d" % (exp_dir, epoch))
    torch.save(optimizer.state_dict(), "%s/models/optim_states.%d" % (exp_dir, epoch))

    # Update best accuracy and epoch and overwrite the model files
    acc = (recalls['A_r10'] + recalls['I_r10']) / 2  
    if acc > best_acc:
      best_acc = acc
      best_epoch = epoch
      shutil.copy("%s/models/audio_model.%d.pth" % (exp_dir, epoch),
                  "%s/models/best_audio_model.pth" % (exp_dir))
      shutil.copy("%s/models/image_model.%d.pth" % (exp_dir, epoch),
                  "%s/models/best_image_model.pth" % (exp_dir))

    _save_progress()
    epoch += 1
      

def validate(audio_model, image_model, val_loader, args):
  # Assign the models to devices
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  batch_time = AverageMeter()

  if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)
  
  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)

  audio_model = audio_model.to(device)
  image_model = image_model.to(device)
  audio_model.eval()
  image_model.eval()

  # Retrieval
  n = val_loader.dataset.__len__()
  audio_embeddings = []
  image_embeddings = []
  matchmaps = {}
  nframes_all = []
  end = time.time()
  with torch.no_grad(): 
    for i, (image_input, audio_input, nframes) in enumerate(val_loader):
      if DEBUG:
        print(audio_input.size())
      audio_output = audio_model(audio_input)
      image_output = image_model(image_input)
      
      image_output = image_output.to('cpu').detach()
      audio_output = audio_output.to('cpu').detach()
      if args.save_matchmap:
        matchmap = computeMatchmap(image_output, audio_output)
        matchmaps[str(i)] = matchmap

      audio_embeddings.append(audio_output)
      image_embeddings.append(image_output)
      pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
      nframes.div_(pooling_ratio)
      nframes_all.append(nframes.cpu())

      batch_time.update(time.time() - end)
      end = time.time()
  
    audio_embeddings = torch.cat(audio_embeddings)
    image_embeddings = torch.cat(image_embeddings)
    nframes_all = torch.cat(nframes_all)

  # Save the match maps
  if args.save_matchmap:
    with open('%s/matchmaps/matchmaps.json', 'w') as f:
      json.dump(matchmaps, f) 
  # Compute recalls
  recalls = calc_recalls(image_embeddings, audio_embeddings, nframes_all, args.simtype)
  A_r10 = recalls['A_r10']
  I_r10 = recalls['I_r10']
  A_r5 = recalls['A_r5']
  I_r5 = recalls['I_r5']
  A_r1 = recalls['A_r1']
  I_r1 = recalls['I_r1']

  sys.stdout.flush()
  print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'.format(A_r10=A_r10, I_r10=I_r10, N=n)) #flush=True)
  print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'.format(A_r5=A_r5, I_r5=I_r5, N=n)) #flush=True)
  print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'.format(A_r1=A_r1, I_r1=I_r1, N=n)) #flush=True)

  return recalls
