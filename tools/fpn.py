# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim





_networks_map = {
  'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
               'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
               'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
               'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
               'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
               },
  'Top_resnet50': { 'C1':'top_base/resnet_v1_50/conv1/Relu:0',
                    'C2':'top_base/resnet_v1_50/block1/unit_2/bottleneck_v1',
                    'C3':'top_base/resnet_v1_50/block2/unit_3/bottleneck_v1',
                    'C4':'top_base/resnet_v1_50/block3/unit_5/bottleneck_v1',
                    'C5':'top_base/resnet_v1_50/block4/unit_3/bottleneck_v1',
             },
  'resnet101': {'C1': '', 'C2': '',
                'C3': '', 'C4': '',
                'C5': '',
               }
}
def _extra_conv_arg_scope_with_bn(weight_decay=0.00001,
                     activation_fn=None,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

def _extra_conv_arg_scope(weight_decay=0.00001, activation_fn=None, normalizer_fn=None):

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      padding='SAME',
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,) as arg_sc:
    with slim.arg_scope(
      [slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
          activation_fn=activation_fn,
          normalizer_fn=normalizer_fn) as arg_sc:
          return arg_sc

def build_pyramid(net_name, end_points, bilinear=True):
  """build pyramid features from a typical network,
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """
  pyramid = {}
  if isinstance(net_name, str):
    pyramid_map = _networks_map[net_name]
  else:
    pyramid_map = net_name
  # pyramid['inputs'] = end_points['inputs']
  #arg_scope = _extra_conv_arg_scope()
  arg_scope = _extra_conv_arg_scope_with_bn()
  with tf.variable_scope('pyramid'):
    with slim.arg_scope(arg_scope):
      
      pyramid['P5'] = \
        slim.conv2d(end_points[pyramid_map['C5']], 256, [1, 1], stride=1, scope='C5')
      
      for c in range(4, 1, -1):
        s, s_ = pyramid['P%d'%(c+1)], end_points[pyramid_map['C%d' % (c)]]

        # s_ = slim.conv2d(s_, 256, [3, 3], stride=1, scope='C%d'%c)
        
        up_shape = tf.shape(s_)
        # out_shape = tf.stack((up_shape[1], up_shape[2]))
        # s = slim.conv2d(s, 256, [3, 3], stride=1, scope='C%d'%c)
        s = tf.image.resize_bilinear(s, [up_shape[1], up_shape[2]], name='C%d/upscale'%c)
        s_ = slim.conv2d(s_, 256, [1,1], stride=1, scope='C%d'%c)
        
        s = tf.add(s, s_, name='C%d/addition'%c)
        s = slim.conv2d(s, 256, [3,3], stride=1, scope='C%d/fusion'%c)
        
        pyramid['P%d'%(c)] = s
      
      return pyramid
  

