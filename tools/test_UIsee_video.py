import _init_paths
from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from data.data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target, anchor_filter
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_before_nms, draw_rpn_after_nms, draw_rpn_nms,rpn_nms_generator
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_berfore_nms, draw_rcnn_after_nms_top,draw_rcnn_nms, rcnn_nms_2d
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

import time
import glob
import tensorflow as tf
slim = tf.contrib.slim

from network import *
from tensorflow.python import debug as tf_debug


def load_dummy_datas(index):

    num_frames = []
    rgbs      =[]
    gt_labels =[]
    gt_3dTo2Ds = []
    gt_boxes2d=[]
    rgbs_norm =[]

    rgb   = cv2.imread(ImagesPath+'/%06d.png'%int(index),1).astype(np.float32, copy=False)
    # rgb=rgb[432:,:]
    # rgb_shape = rgb.shape
    # resize_scale=0.6

    # rgb = cv2.resize(rgb,(int(rgb_shape[1]*resize_scale), int(rgb_shape[0]*resize_scale)))
    rgbs_norm0=(rgb-PIXEL_MEANS)/255

    rgbs.append(rgb)
    rgbs_norm.append(rgbs_norm0)

    return  rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, index

is_show=1
# MM_PER_VIEW1 = 120, 30, 70, [1,1,0]
MM_PER_VIEW1 = 180, 70, 60, [1,1,0]#[ 12.0909996 , -1.04700089, -2.03249991]

# train_data_root='/home/users/hhs/4T/datasets/dummy_datas/seg'

ImagesPath = "/home/hhs/4T/datasets/KITTI/object/training/image_2/"
# ImagesPath='/home/hhs/4T/datasets/Last_14000/Raw_Images'

# ImagesPath='/home/hhs/4T/hongsheng/2dTo3D/faster_rcnn/examples/source_sequence/'+ target_sequence
# source_sequence = {0: '2016_0306_110310_227', 1: '03010907_0024', 2:'03010916_0027', 3: '03070855_0046', 4: 'CLIP0121', 5: 'CLIP0233', \
# 6: 'CLIP0238', 7: 'kitti_005', 8: 'kitti_059', 9: 'kitti_064', 10: 'KITTI_Train'}
# target_sequence = source_sequence[8]
# save_path = '/home/hhs/4T/hongsheng/2dTo3D/faster_rcnn/examples/result_sequence/'+'result_crop_'+ target_sequence
# save_path2d = '/home/hhs/4T/hongsheng/2dTo3D/faster_rcnn/examples/result_sequence/'+'2d_'+ target_sequence
# empty(save_path)
# makedirs(save_path)
# empty(save_path2d)
# makedirs(save_path2d)


def run_test():
    CFG.KEEPPROBS = 1
    # output dir, etc
    out_dir = './outputs'
    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    log = Logger(out_dir+'/log/log_%s.txt'%(time.strftime('%Y-%m-%d %H:%M:%S')),mode='a')

    files_list=glob.glob(ImagesPath+"/*.png")
    index=np.array([int(file_index.strip().split('/')[-1].split('.')[0]) for file_index in files_list ])
    # index=np.array([int(file_index.strip().split('/')[-1].split('.')[0].split('_')[-1]) for file_index in files_list ])
    index=sorted(index)
    print('len(index):%d'%len(index))
    num_frames=len(index)
    # pdb.set_trace()


    if 1:
        ratios_rgb=np.array([0.5,1,2], dtype=np.float32)
        scales_rgb=np.array([0.5,1,2,4,5],   dtype=np.float32)
        bases_rgb = make_bases(
            base_size = 48,
            ratios=ratios_rgb,
            scales=scales_rgb
        )

        num_bases_rgb = len(bases_rgb)
        stride = 8

        rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, image_index = load_dummy_datas(index[10])    
        rgb_shape   = rgbs[0].shape
        rgb_feature_shape = ((rgb_shape[0]-1)//stride+1, (rgb_shape[1]-1)//stride+1)
        out_shape=(2,2)

    # set anchor boxes
    num_class = 2 #incude background
    anchors_rgb, inside_inds_rgb =  make_anchors(bases_rgb, stride, rgb_shape[0:2], rgb_feature_shape[0:2])
    print ('out_shape=%s'%str(out_shape))
    print ('num_frames=%d'%num_frames)


    #load model ####################################################################################################
    rgb_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors_rgb'    )
    rgb_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds_rgb')

    rgb_images   = tf.placeholder(shape=[None, None, None, 3 ], dtype=tf.float32, name='rgb'  )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

    rgb_features, rgb_scores, rgb_probs, rgb_deltas= \
        top_feature_net(rgb_images, rgb_anchors, rgb_inside_inds, num_bases_rgb)

    fuse_scores, fuse_probs, fuse_deltas, fuse_deltas_3dTo2D = \
        fusion_net(
            ( [rgb_features,     rgb_rois,     7,7,1./(1*stride)],),num_class, out_shape) #<todo>  add non max suppression

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()  
        # saver.restore(sess, './outputs/check_points/snap_2D_pretrain.ckpt')
        # saver.restore(sess, './outputs/check_points/snap_2dTo3D_val_120000.ckpt')
        saver.restore(sess, './outputs/check_points/snap_2dTo3D__data_augmentation090000trainval.ckpt')
        # 
        # # pdb.set_trace()
        # var_lt_res=[v for v in tf.global_variables() if not v.name.startswith('fuse/3D')]
        # saver_0=tf.train.Saver(var_lt_res) 
        # saver_0.restore(sess, './outputs/check_points/snap_2D_pretrain.ckpt')

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0

        for iter in range(num_frames):
            start_time=time.time()
            # iter=iter+20
            print('Processing Img: %d  %s'%(iter, index[iter]))
            rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, image_index = load_dummy_datas(index[iter])
            idx=0

            rgb_shape   = rgbs[idx].shape
            # top_img=top_imgs[idx]

            batch_rgb_images    = rgbs_norm[idx].reshape(1,*rgb_shape)

            # batch_gt_labels    = gt_labels[idx]
            # batch_gt_3dTo2Ds   = gt_3dTo2Ds[idx]
            # batch_gt_boxes2d   = gt_boxes2d[idx]
            # if len(batch_gt_labels)==0:
            #     # idx=idx+1
            #     # pdb.set_trace()
            #     cv2.waitKey(0)
                # continue  
            ## run propsal generation ------------
            fd1={

                rgb_images:      batch_rgb_images,
                rgb_anchors:     anchors_rgb,
                rgb_inside_inds: inside_inds_rgb,

                IS_TRAIN_PHASE:  False
            }
            batch_rgb_probs, batch_deltas, batch_rgb_features = sess.run([rgb_probs, rgb_deltas, rgb_features],fd1)

            rgb_feature_shape = ((rgb_shape[0]-1)//stride+1, (rgb_shape[1]-1)//stride+1)
            anchors_rgb, inside_inds_rgb =  make_anchors(bases_rgb, stride, rgb_shape[0:2], rgb_feature_shape[0:2])
                        # pdb.set_trace()
            nms_pre_topn_=2000
            nms_post_topn_=300  
            img_scale=1
            rpn_nms = rpn_nms_generator(stride, rgb_shape[1], rgb_shape[0], img_scale, nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_)  
            batch_proposals, batch_proposal_scores=rpn_nms(batch_rgb_probs, batch_deltas, anchors_rgb, inside_inds_rgb)  

            ## run classification and regression  -----------

            fd2={
                **fd1,

                rgb_images:      batch_rgb_images,

                rgb_rois:        batch_proposals,

            }
            # batch_top_probs,  batch_top_deltas  =  sess.run([ top_probs,  top_deltas  ],fd2)
            batch_fuse_probs, batch_fuse_deltas, batch_fuse_deltas_3dTo2D =  sess.run([ fuse_probs, fuse_deltas, fuse_deltas_3dTo2D ],fd2)
            
            probs, boxes2d, projections = rcnn_nms_2d(batch_fuse_probs, batch_fuse_deltas, batch_proposals, batch_fuse_deltas_3dTo2D, threshold=0.3)
            speed=time.time()-start_time
            print('speed: %0.4fs'%speed)
            # pdb.set_trace()
            # debug: ------------------------------------
            if is_show == 1:

                rgb=rgbs[idx]
              
                img_rpn_nms = draw_rpn_nms(rgb, batch_proposals, batch_proposal_scores)
                # img_gt     = draw_rpn_gt(rgb, batch_gt_boxes2d, batch_gt_labels)
                # imshow('img_gt',img_gt)
                # imshow('img_rpn_nms',img_rpn_nms)

                img_rcnn_nms = draw_rgb_projections(rgb, projections, color=(0,0,255), thickness=1)
                img_rgb_2d_detection = draw_boxes(rgb, boxes2d, color=(255,0,255), thickness=1)
                imshow('draw_rcnn_nms',img_rcnn_nms)
                # imshow('img_rgb_2d_detection',img_rgb_2d_detection)
                # cv2.imwrite(save_path2d+'/%05d.png'%index[iter],img_rgb_2d_detection)
                # cv2.imwrite(save_path+'/%05d.png'%index[iter],img_rcnn_nms)

                cv2.waitKey(50)
                # plt.pause(0.25)
                # mlab.clf(mfig)

## main function ##########################################################################


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_test()
