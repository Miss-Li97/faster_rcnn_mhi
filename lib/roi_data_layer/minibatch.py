# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import torchvision.transforms as tt
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob,mhi_list_to_blob
import pdb
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  #lhy:从_get_image_blob中获取MHI_data
  im_blob, im_scales,MHI_data = _get_image_blob(roidb, random_scale_inds)
  #把MHI_data写进blobs中
  blobs = {'data': im_blob,'MHI':MHI_data}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)

  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  processed_mhis=[]
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])#返回图片的地址
    #print(roidb[i]['image'])
    #lhy:获取mhi的地址，并读取图片得到MHI_data
    MHI_path=roidb[i]['image'].replace("JPEGImages","MHI")
    #print(MHI_path)
    MHI_data=imread(MHI_path)
    # im=tt.Compose([
    #   tt.Resize((660,1280)),
    #   tt.ToTensor()
    # ] )
    # MHI_data=tt.Compose([
    #   tt.Resize((660,1280)),
    #   tt.ToTensor()
    # ] )
    # print(im)#(360,450,3)
    # print(MHI_data)#(360,450)
    if len(im.shape) == 2 :
      im = im[:,:,np.newaxis]#在np.newaxis所在的位置增加一个维度
      im = np.concatenate((im,im,im), axis=2)#数组拼接
    # if len(MHI_data.shape) == 2:
    #   MHI_data = MHI_data[:, :, np.newaxis]  # 在np.newaxis所在的位置增加一个维度
    #   MHI_data = np.concatenate((MHI_data, MHI_data, MHI_data), axis=2)  # 数组拼接
    #   print(MHI_data.shape)#(160,208,3)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]#修改最后一维中rgb的值为gbr
    # MHI_data=MHI_data[:,:,::-1]
    # print("00000")
    # print(MHI_data.shape)
    # print("!!!!!!!!!!!")
    # MHI_data = MHI_data[:, ::-1, :]
    # print(MHI_data.shape)
    # print("im3")
    # print(im)
    # print(im.shape)

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      #lhy
      MHI_data = MHI_data[:, ::-1]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
     #lhy
    im, im_scale,MHI_data = prep_im_for_blob(im, MHI_data,cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)#PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    im_scales.append(im_scale)
    processed_ims.append(im)
    #lhy
    processed_mhis.append(MHI_data)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  #lhy
  MHI_data=mhi_list_to_blob(processed_mhis)

  return blob, im_scales,MHI_data#lhy
