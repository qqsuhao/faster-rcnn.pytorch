# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'   # * 预训练权值文件路径
    self.dout_base_model = 512                                  # * 骨干网输出的维度（应该是特征图的通道数）
    self.pretrained = pretrained                                # * 是否预训练
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()                                                                            # * 创建VGG16
    if self.pretrained:                                                                             # * 加载预训练权值
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])        # * vgg.py里边可以看到，官方VGG的结构是feature-->avgpool-->reshape-->classifier
                                                                                        # * 其中classifier是一个全连接层，feature就是我们要用到的骨干网
    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False        # * 锁住VGG16骨干网前10层的权值，禁止梯度更新

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)                 # * RCNN+全连接层输出各个类别的概率

    if self.class_agnostic:                                               # * 这里可以看到该参数的作用：是否对每个类别进行目标框预测
      self.RCNN_bbox_pred = nn.Linear(4096, 4)                            # * RCNN+全连接层输出预测框的位置
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):         # * 结合faster_rcnn.py中的代码，这里pool5应该表示roi_pooling的输出
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

