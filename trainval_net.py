# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet #,mhi_resnet
from model.faster_rcnn.MHI import ResNet18

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=2, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=4, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=True, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=3, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=56642, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()#torch.arange(0,3)-->([0,1,2])
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':
  n = 0
  with open('loss_4'
            ''
            ''
            '.txt', 'w+') as loss_w:

      args = parse_args()

      print('Called with args:')
      print(args)

      if args.dataset == "pascal_voc":
          args.imdb_name = "voc_2007_trainval"
          args.imdbval_name = "voc_2007_test"
          args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      elif args.dataset == "pascal_voc_0712":
          args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
          args.imdbval_name = "voc_2007_test"
          args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      elif args.dataset == "coco":
          args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
          args.imdbval_name = "coco_2014_minival"
          args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      elif args.dataset == "imagenet":
          args.imdb_name = "imagenet_train"
          args.imdbval_name = "imagenet_val"
          args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
      elif args.dataset == "vg":
          # train sizes: train, smalltrain, minitrain
          # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
          args.imdb_name = "vg_150-50-50_minitrain"
          args.imdbval_name = "vg_150-50-50_minival"
          args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

      args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

      if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
      if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

      print('Using config:')
      pprint.pprint(cfg) #pprint模块 提供了打印出任何python数据结构类和方法。
      np.random.seed(cfg.RNG_SEED)

      #torch.backends.cudnn.benchmark = True
      if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

      # train set
      # -- Note: Use validation set and disable the flipped to enable faster loading.
      cfg.TRAIN.USE_FLIPPED = True
      cfg.USE_GPU_NMS = args.cuda
      imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)#voc_2007_trainval
      '''
      imdb 是一个pascal_voc的类
      roidb['boxes' 、'gt_classes'、'gt_overlaps'、'flipped','image'、'width'、'height'、'max_classes'、'max_overlaps']
      ratio_list
      ratio_index
      '''
      train_size = len(roidb)

      print('{:d} roidb entries'.format(len(roidb)))

      output_dir = args.save_dir + "/" + args.net + "/" + args.dataset #存储模型参数的路径 /models/res101/pascal_voc
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #数据读取
      sampler_batch = sampler(train_size, args.batch_size)

      dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

      dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                sampler=sampler_batch, num_workers=args.num_workers )

    #数据初始化
      # initilize the tensor holder here.
      im_data = torch.FloatTensor(1)
      im_info = torch.FloatTensor(1)
      num_boxes = torch.LongTensor(1)
      gt_boxes = torch.FloatTensor(1)
      MHI_data=torch.FloatTensor(1)

      # ship to cuda
      if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        MHI_data=MHI_data.cuda()

      # make variable
      im_data = Variable(im_data)
      im_info = Variable(im_info)
      num_boxes = Variable(num_boxes)
      gt_boxes = Variable(gt_boxes)
      MHI_data=Variable(MHI_data)

      if args.cuda:
        cfg.CUDA = True

      # initilize the network here. 初始化模型
      # use_cuda = torch.cuda.is_available()
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
      elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
      elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
      elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
      else:
        print("network is not defined")
      # if use_cuda:
      #     fasterRCNN.cuda()
      # # net = net()
      # if torch.cuda.device_count() > 1:
      #     net = nn.DataParallel(fasterRCNN, device_ids=[0, 1])
      #
      # fasterRCNN.to(device)
      fasterRCNN.create_architecture()
      #lhy 得到reanet18网络
      net = ResNet18().to(device)

      # lhy:选择net的优化器（net用来提取mhi_feat）
      #opt = torch.optim.Adam(net.parameters(), lr=1e-3)


      lr = cfg.TRAIN.LEARNING_RATE #0.001
      lr = args.lr #0.001
      #tr_momentum = cfg.TRAIN.MOMENTUM
      #tr_momentum = args.momentum
    #加载模型参数
      params = []
      '''
        RCNN_top.0.2.conv2.weight
        Parameter containing:
        tensor([[[[ 1.4023e-02, -1.5075e-04, -2.7093e-03],....
      '''
      for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:#value是否需要更新即是否可以求导
          if 'bias' in key:#eg: RCNN_top.0.2.bn1.bias
            params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]##TRAIN.DOUBLE_BIAS = True C.TRAIN.BIAS_DECAY = False TRAIN.WEIGHT_DECAY=0.0005
          else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    #定义优化器
      if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

      elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

      if args.cuda:
        fasterRCNN.cuda()


      if args.resume:
        load_name = os.path.join(output_dir,
          'faster_rcnn_mhi_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
          cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

      if args.mGPUs:
        #fasterRCNN = nn.DataParallel(fasterRCNN,device_ids=[0,1])
        fasterRCNN = nn.DataParallel(fasterRCNN)

    #数据迭代
      iters_per_epoch = int(train_size / args.batch_size)

      if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

      for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        #lhy
        net.train()

        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
          try:
              data = next(data_iter)
          except RuntimeError as e:
              if 'invalid argument 0' in str(e):
                   n+=1
                   print('Got 1000 and 800 in dimension 2 at /pytorch/aten/src/TH/generic/THTensorMath.c:3586')
                   step+=1
                   print(n)
              else:
                  raise e


          im_data.data.resize_(data[0].size()).copy_(data[0])
          im_info.data.resize_(data[1].size()).copy_(data[1])
          gt_boxes.data.resize_(data[2].size()).copy_(data[2])
          num_boxes.data.resize_(data[3].size()).copy_(data[3])
          #lhy
          MHI_data.data.resize_(data[4].size()).copy_(data[4])

          # #lhy:把MHI_data从单通道变为多通道
          # MHI_data = np.concatenate((MHI_data, MHI_data, MHI_data), axis=1)  # 数组拼接
          MHI_data = torch.tensor(MHI_data)
          MHI_data=MHI_data.to(device)
          #lhy;提取运动历史图像的特征
          MHI_feat=net(MHI_data)
          MHI_feat=MHI_feat.detach()

          #lhy:将运动历史图像的特征传入faster rcnn.py中，和视频帧的特征进行拼接
          fasterRCNN.zero_grad()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,MHI_feat)
          # rois, cls_prob, bbox_pred, \
          # rpn_loss_cls, rpn_loss_box, \
          # RCNN_loss_cls, RCNN_loss_bbox, \
          # rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

          loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
          loss_temp += loss.item()

          # backward
          optimizer.zero_grad()
          #opt.zero_grad()
          loss.backward()
          if args.net == "vgg16":
              clip_gradient(fasterRCNN, 10.)
          optimizer.step()
          #opt.step()

          if step % args.disp_interval == 0:
            end = time.time()
            if step > 0:
              loss_temp /= (args.disp_interval + 1)

            if args.mGPUs:
            #if True:
              loss_rpn_cls = rpn_loss_cls.mean().item()
              loss_rpn_box = rpn_loss_box.mean().item()
              loss_rcnn_cls = RCNN_loss_cls.mean().item()
              loss_rcnn_box = RCNN_loss_bbox.mean().item()
              fg_cnt = torch.sum(rois_label.data.ne(0))
              bg_cnt = rois_label.data.numel() - fg_cnt
            else:
              loss_rpn_cls = rpn_loss_cls.item()
              loss_rpn_box = rpn_loss_box.item()
              loss_rcnn_cls = RCNN_loss_cls.item()
              loss_rcnn_box = RCNN_loss_bbox.item()
              fg_cnt = torch.sum(rois_label.data.ne(0))
              bg_cnt = rois_label.data.numel() - fg_cnt

            print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                    % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
            print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
            print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
            loss_w.write(
                "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e,fg/bg=(%d/%d), time cost: %f,rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f\n" \
                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, fg_cnt, bg_cnt, end - start, loss_rpn_cls,loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
            if args.use_tfboard:
              info = {
                'loss': loss_temp,
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_box': loss_rpn_box,
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_box': loss_rcnn_box
              }
              logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

            loss_temp = 0
            start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_mhi_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

      if args.use_tfboard:
        logger.close()
