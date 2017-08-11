from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling
import pdb
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import vgg
from fpn import build_pyramid

keep_prob=0.5
# nms_pre_topn_=5000
# nms_post_topn_=2000

# keep_prob=1
# nms_pre_topn_=2000
# nms_post_topn_=300

is_training=True

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

  #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
  # with tf.variable_scope('nms_top') as scope:    #non-max
  #   batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
  #   img_scale = 1
  #   # pdb.set_trace()
  #   rois, roi_scores,proposals_z, inside_inds_nms = tf_rpn_nms( probs, deltas, anchors, inds_inside,
  #                                    stride, img_width, img_height, img_scale, deltasZ,
  #                                    nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_,
  #                                    name ='nms')
  feature = block
  # feature = tf.stop_gradient(block)
  # scores = tf.stop_gradient(scores)
  # probs = tf.stop_gradient(probs)
  # deltas = tf.stop_gradient(deltas)
  return feature, scores, probs, deltas#, rois, roi_scores,deltasZ, proposals_z, inside_inds_nms


# def top_feature_net(input, anchors, inds_inside, num_bases):
#   stride=4
#   with tf.variable_scope("top_base") as sc:
#     arg_scope = resnet_v1.resnet_arg_scope(is_training=True)
#     with slim.arg_scope(arg_scope):
#       net, end_points = resnet_v1.resnet_v1_50(input, None, global_pool=False)
#     with tf.variable_scope("top_rgb_up") as sc:
#         pyramid=build_pyramid('Top_resnet50', end_points, bilinear=True)
#         block=pyramid['P2']  
#   with tf.variable_scope('rpn_top') as scope:
#     up      = conv2d_relu(block, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#     scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
#     probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
#     deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')
#     deltasZ  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='deltaZ')

#   #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
#   with tf.variable_scope('nms_top') as scope:    #non-max
#     batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
#     img_scale = 1
#     # pdb.set_trace()
#     rois, roi_scores,proposals_z = tf_rpn_nms( probs, deltas, anchors, inds_inside,
#                                      stride, img_width, img_height, img_scale, deltasZ,
#                                      nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_,
#                                      name ='nms')
#   feature = block
#   return feature, scores, probs, deltas, rois, roi_scores,deltasZ, proposals_z


# def top_feature_net(input_, anchors, inds_inside, num_bases):
#   stride=4
#     # arg_scope = resnet_v1.resnet_arg_scope(weight_decay=0.0)
#     # with slim.arg_scope(arg_scope) :
#   with slim.arg_scope(vgg.vgg_arg_scope()):
#     block5, end_points = vgg.vgg_16(input_)
#     block3 = end_points['vgg_16/conv3/conv3_3']
#     block4 = end_points['vgg_16/conv4/conv4_3']
#   with tf.variable_scope("top_base") as sc:
#     block5_   = conv2d_relu(block5, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='5')
#     up_shape = tf.shape(block3)
#     up5 = tf.image.resize_bilinear(block5_, [up_shape[1], up_shape[2]], name='up5')
#     block4_   = conv2d_relu(block4, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4')
#     up4 = tf.image.resize_bilinear(block4_, [up_shape[1], up_shape[2]], name='up4')
#     up_      =tf.add(up4, up5, name="up_add")
#     block3_   = conv2d_relu(block3, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='3')
#     up       = tf.add(up_,block3_, name='fpn_up_')
#     block    = conv2d_relu(up, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='fpn_up')
#     tf.summary.histogram('rpn_top_block', block) 
#     with tf.variable_scope('top') as scope:
#       up      = conv2d_relu(block, num_kernels=256, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#       scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
#       probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
#       deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')
  
#     #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
#   # with tf.variable_scope('top-nms') as scope:    #non-max

#   #   batch_size, img_height, img_width, img_channel = input_.get_shape().as_list()
#   #   img_scale = 1
#   #   # pdb.set_trace()
#   #   rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
#   #                                    stride, img_width, img_height, img_scale,
#   #                                    nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_,
#   #                                     name ='nms')
  
#     #<todo> feature = upsample2d(block, factor = 4,  ...)
#     feature = block
    
#       # print ('top: scale=%f, stride=%d'%(1./stride, stride))
#   return feature, scores, probs, deltas

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
        # roi_features=flatten(roi_features)

        # roi_features_ = tf.stop_gradient(roi_features)

        with tf.variable_scope('fuse-block-1-%d'%n):
          tf.summary.histogram('fuse-block_input_%d'%n, roi_features)
          

          feature   = conv2d_bn_relu(roi_features, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='1%d'%n)
          feature   = conv2d_bn_relu(feature, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2%d'%n)
          feature   = conv2d_bn_relu(feature, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3%d'%n)
          feature   = conv2d_bn_relu(feature, num_kernels=256, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='4%d'%n)
          roi_features = maxpool(feature, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='fuse_maxpooling' )
          roi_features=flatten(roi_features)

          block = linear_bn_relu(roi_features, num_hiddens=1024, name='1')#512, so small?
          tf.summary.histogram('fuse-block1_%d'%n, block)
          block = tf.nn.dropout(block, keep_prob, name='drop1')
  
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
    block = tf.nn.dropout(block, keep_prob, name='drop4')
    with tf.variable_scope('2D') as sc:
      dim = np.product([*out_shape])
      scores_3d  = linear(block, num_hiddens=num_class,     name='score')
      probs_3d   = tf.nn.softmax (scores_3d, name='prob')
      deltas_3d  = linear(block, num_hiddens=dim*num_class, name='box')
      deltas_3d  = tf.reshape(deltas_3d,(-1,num_class,*out_shape))
    with tf.variable_scope('3D') as sc_:
      block3D = linear_bn_relu(roi_features, num_hiddens=512, name='1')#512, so small?
      block3D_1 = tf.nn.dropout(block3D, keep_prob, name='drop1')
      block = linear_bn_relu(block3D_1, num_hiddens=512, name='3D')
      # block = tf.nn.dropout(block, keep_prob, name='drop4')
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

