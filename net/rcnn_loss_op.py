from net.common import *

def rcnn_loss(scores, deltas, rcnn_labels, rcnn_targets):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1


    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + rcnn_labels
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas', rcnn_deltas)
    rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

    index_True=tf.equal(rcnn_labels, tf.ones_like(rcnn_labels))
    rcnn_deltas_=tf.boolean_mask(rcnn_deltas,  index_True)
    rcnn_targets_=tf.boolean_mask(rcnn_targets,  index_True)   

    rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_, rcnn_targets_, sigma=3.0)
    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    return rcnn_cls_loss, rcnn_reg_loss


def rcnn_loss_3dTo2D(scores, deltas, rcnn_labels, rcnn_targets, deltas_3dTo2D, rcnn_targets_3dTo2D):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1

    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + rcnn_labels
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas', rcnn_deltas)
    rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

    index_True=tf.equal(rcnn_labels, tf.ones_like(rcnn_labels))
    rcnn_deltas_=tf.boolean_mask(rcnn_deltas,  index_True)
    rcnn_targets_=tf.boolean_mask(rcnn_targets,  index_True)  

    rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_, rcnn_targets_, sigma=5.0)
    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    dim = np.prod(deltas_3dTo2D.get_shape().as_list()[1:])//num_class
    deltas_3dTo2D      = tf.reshape(deltas_3dTo2D,[-1, dim])
    rcnn_deltas_3dTo2D  = tf.gather(deltas_3dTo2D,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas_2d', rcnn_deltas_3dTo2D)
    rcnn_targets_3dTo2D =  tf.reshape(rcnn_targets_3dTo2D,[-1, dim])

    rcnn_deltas_3dTo2D_=tf.boolean_mask(rcnn_deltas_3dTo2D,  index_True)
    rcnn_targets_3dTo2D_=tf.boolean_mask(rcnn_targets_3dTo2D,  index_True) 

    rcnn_smooth_l1_3dTo2D = modified_smooth_l1(rcnn_deltas_3dTo2D_, rcnn_targets_3dTo2D_, sigma=5.0)
    rcnn_reg_loss_3dTo2D  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1_3dTo2D, axis=1))


    return rcnn_cls_loss, rcnn_reg_loss, rcnn_reg_loss_3dTo2D#, rcnn_pos_inds#


























def rcnn_loss_2d(scores, deltas, rcnn_labels, rcnn_targets, deltas_2d, rcnn_targets_2d,rcnn_pos_inds):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1

    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + rcnn_labels
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas', rcnn_deltas)
    rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

    index_True_0=tf.equal(rcnn_labels, tf.ones_like(rcnn_labels))
    # rcnn_deltas_=tf.boolean_mask(rcnn_deltas,  index_True)
    # rcnn_targets_=tf.boolean_mask(rcnn_targets,  index_True)  

    # index_True=tf.equal(rcnn_labels, 1)
    # rcnn_pos_inds = tf.where(index_True)
    rcnn_deltas_=tf.gather(rcnn_deltas,  rcnn_pos_inds)
    rcnn_targets_=tf.gather(rcnn_targets,  rcnn_pos_inds)  

    rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_, rcnn_targets_, sigma=3.0)
    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    dim = np.prod(deltas_2d.get_shape().as_list()[1:])//num_class
    deltas1_2d      = tf.reshape(deltas_2d,[-1, dim])
    rcnn_deltas_2d  = tf.gather(deltas1_2d,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas_2d', rcnn_deltas_2d)
    rcnn_targets_2d =  tf.reshape(rcnn_targets_2d,[-1, dim])

    rcnn_deltas_2d_=tf.gather(rcnn_deltas_2d,  rcnn_pos_inds)
    rcnn_targets_2d_=tf.gather(rcnn_targets_2d,  rcnn_pos_inds) 

    rcnn_smooth_l1_2d = modified_smooth_l1(rcnn_deltas_2d_, rcnn_targets_2d_, sigma=3.0)
    rcnn_reg_loss_2d  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1_2d, axis=1))



    return rcnn_cls_loss, rcnn_reg_loss, rcnn_reg_loss_2d#, rcnn_pos_inds#


# def rcnn_loss_2d(scores, deltas, rcnn_labels, rcnn_targets, deltas_2d, rcnn_targets_2d):

#     def modified_smooth_l1( deltas, targets, sigma=3.0):
#         '''
#             ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
#             SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
#                           |x| - 0.5 / sigma^2,    otherwise
#         '''
#         sigma2 = sigma * sigma
#         diffs  =  tf.subtract(deltas, targets)
#         smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

#         smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
#         smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
#         smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
#         smooth_l1 = smooth_l1_add

#         return smooth_l1

#     _, num_class = scores.get_shape().as_list()
#     dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

#     rcnn_scores   = tf.reshape(scores,[-1, num_class])
#     rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

#     num = tf.shape(deltas)[0]
#     idx = tf.range(num)*num_class + rcnn_labels
#     deltas1      = tf.reshape(deltas,[-1, dim])
#     rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
#     tf.summary.histogram('rcnn_deltas', rcnn_deltas)
#     rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

#     index_True=tf.equal(rcnn_labels, tf.ones_like(rcnn_labels))
#     rcnn_deltas_=tf.boolean_mask(rcnn_deltas,  index_True)
#     rcnn_targets_=tf.boolean_mask(rcnn_targets,  index_True)   

#     rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_, rcnn_targets_, sigma=3.0)
#     rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))



#     dim = np.prod(deltas_2d.get_shape().as_list()[1:])//num_class
#     deltas1_2d      = tf.reshape(deltas_2d,[-1, dim])
#     rcnn_deltas_2d  = tf.gather(deltas1_2d,  idx)  # remove ignore label
#     tf.summary.histogram('rcnn_deltas_2d', rcnn_deltas_2d)
#     rcnn_targets_2d =  tf.reshape(rcnn_targets_2d,[-1, dim])

#     rcnn_deltas_2d_=tf.boolean_mask(rcnn_deltas_2d,  index_True)
#     rcnn_targets_2d_=tf.boolean_mask(rcnn_targets_2d,  index_True) 

#     rcnn_smooth_l1_2d = modified_smooth_l1(rcnn_deltas_2d_, rcnn_targets_2d_, sigma=3.0)
#     rcnn_reg_loss_2d  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1_2d, axis=1))



#     return rcnn_cls_loss, rcnn_reg_loss, rcnn_reg_loss_2d

def rcnn_loss_ohem(scores, deltas, rcnn_labels, rcnn_targets):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1


    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    softmax_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels)
    # rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + rcnn_labels
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
    tf.summary.histogram('rcnn_deltas', rcnn_deltas)
    rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

    index_True=tf.equal(rcnn_labels, tf.ones_like(rcnn_labels))
    rcnn_deltas_=tf.boolean_mask(rcnn_deltas,  index_True)
    rcnn_targets_=tf.boolean_mask(rcnn_targets,  index_True)
    
    
    rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_, rcnn_targets_, sigma=3.0)
    rcnn_smooth_l1=tf.reduce_sum(rcnn_smooth_l1, axis=1)
    # rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    return softmax_loss, rcnn_smooth_l1
