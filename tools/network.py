from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from net.configuration import *
from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling
import pdb
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import vgg
from fpn import build_pyramid



def top_feature_net(input, anchors, inds_inside, num_bases):
  stride=8
  arg_scope = resnet_v1.resnet_arg_scope(is_training=True)
  with slim.arg_scope(arg_scope):
    net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False, output_stride=16)
    block4=end_points['resnet_v1_50/block4/unit_3/bottleneck_v1']
    block3=end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
    block2=end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']
    tf.summary.histogram('top_block4', block4)
    tf.summary.histogram('top_block3', block3)
    tf.summary.histogram('top_block2', block2)
  with tf.variable_scope("top_up") as sc:
    block4_   = conv2d_relu(block4, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4')
    up_shape = tf.shape(block2)
    up4 = tf.image.resize_bilinear(block4_, [up_shape[1], up_shape[2]], name='up4')
    block3_   = conv2d_relu(block3, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='3')
    up3 = tf.image.resize_bilinear(block3_, [up_shape[1], up_shape[2]], name='up3')
    block2_   = conv2d_relu(block2, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='2')
    # up2     = upsample2d(block2_, factor = 2, has_bias=True, trainable=True, name='up2')
    up_34      =tf.add(up4, up3, name="up_add_3_4")
    up      =tf.add(up_34, block2_, name="up_add_3_4_2")
    block    = conv2d_relu(up, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='rgb_ft')
  with tf.variable_scope('rpn_top') as scope:
    up      = conv2d_relu(block, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
    scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
    probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
    deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')
    deltasZ  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='deltaZ')

  feature = block
  return feature, scores, probs, deltas#, rois, roi_scores,deltasZ, proposals_z, inside_inds_nms



#------------------------------------------------------------------------------
def fusion_net(feature_list, num_class, out_shape=(2,2)):
  num=len(feature_list)

  input = None
  with tf.variable_scope('fuse-input') as scope:
    for n in range(num):
        feature     = feature_list[n][0]
        roi         = feature_list[n][1]
        pool_height = feature_list[n][2]
        pool_width  = feature_list[n][3]
        pool_scale  = feature_list[n][4]
        if (pool_height==0 or pool_width==0): continue
        # feature   = conv2d_bn_relu(feature, num_kernels=512, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='%d'%n)
        roi_features,  roi_idxs = tf_roipooling(feature,roi, pool_height, pool_width, pool_scale, name='%d/pool'%n)
        # pdb.set_trace()
        roi_features=flatten(roi_features)
        # roi_features_ = tf.stop_gradient(roi_features)

        with tf.variable_scope('fuse-block-1-%d'%n):
          tf.summary.histogram('fuse-block_input_%d'%n, roi_features)
          block = linear_bn_relu(roi_features, num_hiddens=2048, name='1')#512, so small?
          tf.summary.histogram('fuse-block1_%d'%n, block)
          block = tf.nn.dropout(block, CFG.KEEPPROBS , name='drop1')
  
        if input is None:
            input = block
        else:
            # input = concat([input,block], axis=1, name='%d/cat'%n)
            input = tf.add(input,block, name ='fuse_feature')
    # input_ = tf.stop_gradient(input)
  input_ = input
  #include background class
  with tf.variable_scope('fuse') as scope:
    block = linear_bn_relu(input_, num_hiddens=512, name='4')#512, so small?
    # block = tf.stop_gradient(block)
    block = tf.nn.dropout(block, CFG.KEEPPROBS , name='drop4')
    with tf.variable_scope('2D') as sc:
      dim = np.product([*out_shape])
      scores_3d  = linear(block, num_hiddens=num_class,     name='score')
      probs_3d   = tf.nn.softmax (scores_3d, name='prob')
      deltas_3d  = linear(block, num_hiddens=dim*num_class, name='box')
      deltas_3d  = tf.reshape(deltas_3d,(-1,num_class,*out_shape))
    with tf.variable_scope('3D') as sc_:
      block3D = linear_bn_relu(roi_features, num_hiddens=2048, name='1')#512, so small?
      block3D_1 = tf.nn.dropout(block3D, CFG.KEEPPROBS , name='drop1')
      block = linear_bn_relu(block3D_1, num_hiddens=512, name='3D')
      # block = tf.nn.dropout(block, CFG.KEEPPROBS , name='drop4')
      dim = np.product(16)
      deltas_2d  = linear(block, num_hiddens=dim*num_class, name='box')
      deltas_2d  = tf.reshape(deltas_2d,(-1,num_class,16))
    # scores_3d = tf.stop_gradient(scores_3d)
    # probs_3d = tf.stop_gradient(probs_3d)
    # deltas_3d = tf.stop_gradient(deltas_3d)


  return  scores_3d, probs_3d, deltas_3d, deltas_2d


# main ###########################################################################
# to start in tensorboard:
#    /opt/anaconda3/bin
#    ./python tensorboard --logdir /root/share/out/didi/tf
#     http://http://localhost:6006/    

