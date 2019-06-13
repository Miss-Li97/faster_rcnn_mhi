"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

'''
roidb是一个列表，列表的长度是读取的图片文件个数的两倍（注意，经过了图片翻转），
列表中的每一个元素都是一个字典，而且字典的key包括：
'boxes' 、'gt_classes'、'gt_overlaps'、'flipped'（原始图片取值为：False，翻转之后的图片取值为：True）、'image'、'width'、'height'、'max_classes'、'max_overlaps'。

'''
def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    cache_file = os.path.join(imdb.cache_path, imdb.name + '_30_sizes.pkl')
    if os.path.exists(cache_file):
      print('Image sizes loaded from %s' % cache_file)
      with open(cache_file, 'rb') as f:
        sizes = pickle.load(f) #sizes=gt_roidb
    else:
      print('Extracting image sizes... (It may take long time)')
      sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                for i in range(imdb.num_images)]
      with open(cache_file, 'wb') as f:
        pickle.dump(sizes, f)
      print('Done!!')
         
  for i in range(len(imdb.image_index)):#读取trainval.txt文件
    roidb[i]['img_id'] = imdb.image_id_at(i) # i
    roidb[i]['image'] = imdb.image_path_at(i) # 获取trainval.txt文件第i行中图片的路径，例如：ILSVRC2015_train_00341000/000266
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]#每张图片的宽度
      roidb[i]['height'] = sizes[i][1]#每张图片的高度
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)# 按照行求最大值，对于gt_box来说结果为np.ones(物体个数)，是一个：一维向量，这里因为只有gt_box，所以取值为1
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)#每张图片中包含的物体的类别号（是一个：一维向量）
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):#imdb_names=voc_2007_trainval
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    # if cfg.TRAIN.USE_FLIPPED: #TRAIN.USE_FLIPPED=true
    #   print('Appending horizontally-flipped training examples...')
    #   imdb.append_flipped_images()
    '''
   imdb.append_flipped_images()函数，这个函数做三件事：第一，将roidb列表做一个copy，然后对copy的元素进行操作：
   （1）修改'boxes'的值（对图片做翻转，那么gt_box的坐标必然会发生变化）
   （2）修改'flipped'的值为：True（代表这张图片是经过翻转之后的图片）
    第二，将修改后的copy数据添加到原来的roidb列表中，这样列表的长度翻倍
    第三，将self._image_index（是一个列表，保存的是图片的名称）复制一份，并和原来的列表拼接在一块，这样self._image_index的长度也翻倍
    '''
      #print('done')

    print('Preparing training data...')

    prepare_roidb(imdb) #这个函数主要是往roidb列表的每一个字典元素中增加更多的key和value：
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')
    '''
    roidb是一个列表，列表的长度是读取的图片文件个数的两倍（注意，经过了图片翻转），
    列表中的每一个元素都是一个字典，而且字典的key包括：
    'boxes' 、'gt_classes'、'gt_overlaps'、'flipped'（原始图片取值为：False，翻转之后的图片取值为：True）、'image'、'width'、'height'、'max_classes'、'max_overlaps'。

    '''
    return imdb.roidb
  
  def get_roidb(imdb_name): #imdb_name=voc_2007_trainval
    imdb = get_imdb(imdb_name)#imdb=pasvoc中init即imdb是一个pascal_voc 类
    print('Loaded dataset `{:s}` for training'.format(imdb.name))#imdb.name=voc_2007_trainval
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)#imdb.set_proposal_method(gt)  #获取bbox的ground_truth信息
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    '''
    roidb是一个列表，列表的长度是读取的图片文件个数的两倍（注意，经过了图片翻转），
列表中的每一个元素都是一个字典，而且字典的key包括：
'boxes' 、'gt_classes'、'gt_overlaps'、'flipped'（原始图片取值为：False，翻转之后的图片取值为：True）、'image'、'width'、'height'、'max_classes'、'max_overlaps'。

'''
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]# s=voc_2007_trainval
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)

  return imdb, roidb, ratio_list, ratio_index
