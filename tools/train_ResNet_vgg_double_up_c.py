import _init_paths
from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from data.data import *      #################

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target, anchor_filter,rpn_target_Z
from net.rcnn_target_op import rcnn_target_3dTo2D

from net.rpn_nms_op     import draw_rpn_nms, draw_rpn,rpn_nms_generator
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms, draw_rcnn,rcnn_nms_2d
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

import mayavi.mlab as mlab

import time
import glob
import tensorflow as tf
slim = tf.contrib.slim

from ResNet50_vgg_double_up_c import *
from tensorflow.python import debug as tf_debug
# os.environ["QT_API"] = "pyqt"

#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017

def load_dummy_datas(index):

    num_frames = []
    rgbs      =[]
    gt_labels =[]
    gt_3dTo2Ds = []
    gt_boxes2d=[]
    rgbs_norm =[]

    # pdb.set_trace()
    if num_frames==[]:
        num_frames=len(index)
        print('num_frames:%d'%num_frames)
    for n in range(num_frames):
        print('processing img:%d,%05d'%(n,int(index[n])))
        try:
            gt_label  = np.load(train_data_root+'/gt_labels/gt_labels_%05d.npy'%int(index[n]))
        except:
            print('No target in this image')
            continue
        rgb   = cv2.imread(kitti_dir+'/image_2/%06d.png'%int(index[n]))
        rgbs_norm0=(rgb-PIXEL_MEANS)/255
 
        gt_label  = np.load(train_data_root+'/gt_labels/gt_labels_%05d.npy'%int(index[n]))
        gt_3dTo2D = np.load(train_data_root+'/gt_3dTo2D/gt_3dTo2D_%05d.npy'%int(index[n]))
        gt_box2d = np.load(train_data_root+'/gt_boxes2d/gt_boxes2d_%05d.npy'%int(index[n]))        


        rgbs.append(rgb)
        gt_labels.append(gt_label)
        gt_3dTo2Ds.append(gt_3dTo2D)
        gt_boxes2d.append(gt_box2d)
        rgbs_norm.append(rgbs_norm0)

        # explore dataset:
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            projections=box3d_to_rgb_projections(gt_box3d)
            rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
            top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)

            imshow('rgb',rgb1)
            imshow('top_image',top_image1)
            azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
            mlab.view(azimuth,elevation,distance,focalpoint)
            mlab.clf(fig)
            draw_lidar(lidar, fig=fig)
            draw_gt_boxes3d(gt_box3d, fig=fig)
            mlab.show(1)
            cv2.waitKey(0)
            mlab.close()
            pass
    # pdb.set_trace()
    # rgbs=np.array(rgbs)
    ##exit(0)
    mlab.close(all=True)
    return  rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, index

# index_list=open(train_data_root+'/trainval.txt')
# index = [ int(i.strip()) for i in index_list]
# print ('length of index : %d'%len(index))
MM_PER_VIEW1 = 180, 70, 30, [1,1,0]
vis=0
# ohem=1
def run_train():

    # output dir, etc
    out_dir = './outputs'
    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    makedirs(out_dir +'/log')
    log = Logger(out_dir+'/log/log_%s.txt'%(time.strftime('%Y-%m-%d %H:%M:%S')),mode='a')
    
    # index=np.load(train_data_root+'/train.npy')
    index_file=open(train_data_root+'/trainval.txt')
    index = [ int(i.strip()) for i in index_file]
    index_file.close()
    index=sorted(index)
    index=np.array(index)
    num_frames = len(index)

     #lidar data -----------------
    if 1:
        ###generate anchor base 
        ratios_rgb=np.array([0.5,1,2], dtype=np.float32)
        scales_rgb=np.array([0.5,1,2,4,5],   dtype=np.float32)
        bases_rgb = make_bases(
            base_size = 48,
            ratios=ratios_rgb,
            scales=scales_rgb
        )

        num_bases_rgb = len(bases_rgb)
        stride = 8
        out_shape=(2,2)

        rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, image_index = load_dummy_datas(index[:3])
        # rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, rgbs_norm, image_index, lidars = load_dummy_datas()

        rgb_shape   = rgbs[0].shape
        rgb_feature_shape = ((rgb_shape[0]-1)//stride+1, (rgb_shape[1]-1)//stride+1)
        # set anchor boxes
        num_class = 2 #incude background
        anchors_rgb, inside_inds_rgb =  make_anchors(bases_rgb, stride, rgb_shape[0:2], rgb_feature_shape[0:2])
        # pdb.set_trace()

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



    #loss ########################################################################################################
    rgb_inds   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='rgb_ind'    )
    rgb_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='rgb_pos_ind')
    rgb_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='rgb_label'  )
    rgb_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='rgb_target' )
    rgb_cls_loss, rgb_reg_loss = rpn_loss(2*rgb_scores, rgb_deltas, rgb_inds, rgb_pos_inds, rgb_labels, rgb_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='fuse_target')
    fuse_targets_3dTo2Ds = tf.placeholder(shape=[None, 16], dtype=tf.float32, name='fuse_target')


    fuse_cls_loss, fuse_reg_loss, fuse_reg_loss_3dTo2D = rcnn_loss_3dTo2D(fuse_scores, fuse_deltas, fuse_labels, fuse_targets, fuse_deltas_3dTo2D, fuse_targets_3dTo2Ds)

    tf.summary.scalar('rcnn_cls_loss', fuse_cls_loss)
    tf.summary.scalar('rcnn_reg_loss', fuse_reg_loss)
    tf.summary.scalar('rcnn_reg_loss_3dTo2D', fuse_reg_loss_3dTo2D)
    tf.summary.scalar('rpn_cls_loss', rgb_cls_loss)
    tf.summary.scalar('rpn_reg_loss', rgb_reg_loss)

    #solver
    l2 = l2_regulariser(decay=0.000005)
    tf.summary.scalar('l2', l2)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.AdamOptimizer(learning_rate)
    solver_step = solver.minimize(2*rgb_cls_loss+1*rgb_reg_loss+2*fuse_cls_loss+1*fuse_reg_loss+0.01*fuse_reg_loss_3dTo2D+l2)
    # 2*rgb_cls_loss+1*rgb_reg_loss+2*fuse_cls_loss+1*fuse_reg_loss+

    max_iter = 200000
    iter_debug=1

    # start training here  #########################################################################################
    log.write('epoch     iter    speed   rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()  
    train_writer = tf.summary.FileWriter( './outputs/tensorboard/V_2dTo3d_Res8_pretrain_stop_gradient_ok2_traintxt_val',
                                      sess.graph)
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver() 

        # saver.restore(sess, './outputs/check_points/snap_2dTo3d_with_2d_pretrained__105000.ckpt') 


        # var_lt_res=[v for v in tf.trainable_variables() if v.name.startswith('resnet_v1_50')]#resnet_v1_50
        # var_lt_res=[v for v in tf.all_variables() if  not v.name.startswith('fuse/3D')]
        var_lt_res=[v for v in tf.all_variables() if  not ('Adam' in v.name)]
        # pdb.set_trace()
        # # var_lt_res.pop(0)
        saver_0=tf.train.Saver(var_lt_res)        
        # saver_0.restore(sess, './outputs/check_points/resnet_v1_50.ckpt')
        # saver_0.restore(sess, './outputs/check_points/snap_2D_pretrain.ckpt')
        saver_0.restore(sess, './outputs/check_points/snap_2dTo3d_with_2d_pretrained_traintxt_010000.ckpt') 
        # pdb.set_trace()
        
        # var_lt_vgg=[v for v in tf.trainable_variables() if v.name.startswith('vgg')]
        # saver_1=tf.train.Saver(var_lt_vgg)
        # saver_1.restore(sess, './outputs/check_points/vgg_16.ckpt')

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        # # rate=0.00005
        frame_range = np.arange(num_frames)
        idx=0
        frame=0
        rate=0.00004
        for iter in range(max_iter):
            epoch=iter//num_frames+1
            # rate=0.001
            start_time=time.time()
            if iter%(num_frames*1)==0:
                idx=0
                frame=0
                count=0
                end_flag=0
                frame_range1 = np.random.permutation(num_frames)
                if np.all(frame_range1==frame_range):
                    raise Exception("Invalid level!", permutation)
                frame_range=frame_range1

            #load 500 samples every 2000 iterations
            freq=int(10)
            if idx%freq==0 :
                count+=idx
                if count%(1*freq)==0:
                    frame+=idx
                    frame_end=min(frame+freq,num_frames)
                    if frame_end==num_frames:
                        end_flag=1
                    # pdb.set_trace()
                    rgbs, gt_labels, gt_3dTo2Ds, gt_boxes2d, rgbs_norm, image_index = load_dummy_datas(index[frame_range[frame:frame_end]])
                idx=0
            if (end_flag==1) and (idx+frame)==num_frames:
                idx=0
            print('processing image : %s'%image_index[idx])

            if (iter+1)%(10000)==0:
                rate=0.7*rate

            rgb_shape   = rgbs[idx].shape
            batch_rgb_images    = rgbs_norm[idx].reshape(1,*rgb_shape)
            # batch_rgb_images    = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            if len(batch_gt_labels)==0:
                idx=idx+1
                continue

            batch_gt_3dTo2Ds   = gt_3dTo2Ds[idx]
            batch_gt_boxes2d   = gt_boxes2d[idx]
            
            ## run propsal generation ------------
            fd1={

                rgb_images:      batch_rgb_images,
                rgb_anchors:     anchors_rgb,
                rgb_inside_inds: inside_inds_rgb,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True,

            }
            batch_rgb_probs, batch_deltas, batch_rgb_features = sess.run([rgb_probs, rgb_deltas, rgb_features],fd1) 

            nms_pre_topn_=5000
            nms_post_topn_=2000  
            img_scale=1
            rpn_nms = rpn_nms_generator(stride, rgb_shape[1], rgb_shape[0], img_scale, nms_thresh=0.7, min_size=stride, nms_pre_topn=nms_pre_topn_, nms_post_topn=nms_post_topn_)  
            batch_proposals, batch_proposal_scores=rpn_nms(batch_rgb_probs, batch_deltas, anchors_rgb, inside_inds_rgb)  

            # pdb.set_trace()
            ## generate  train rois  ------------
            batch_rgb_inds, batch_rgb_pos_inds, batch_rgb_labels, batch_rgb_targets  = \
                rpn_target ( anchors_rgb, inside_inds_rgb, batch_gt_labels,  batch_gt_boxes2d)
            batch_rgb_rois, batch_fuse_labels, batch_fuse_targets2d, batch_fuse_targets_3dTo2Ds = rcnn_target_3dTo2D(batch_proposals, batch_gt_labels, batch_gt_boxes2d, batch_gt_3dTo2Ds, rgb_shape[1], rgb_shape[0])
            # pdb.set_trace()
            #      rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )

            batch_rgb_rois, batch_fuse_labels, batch_fuse_targets2d, batch_fuse_targets_3dTo2Ds = rcnn_target_3dTo2D(batch_proposals, batch_gt_labels, batch_gt_boxes2d, batch_gt_3dTo2Ds, rgb_shape[1], rgb_shape[0])


            print('nums of rcnn batch: %d'%len(batch_rgb_rois))
            ##debug gt generation
            if vis and iter%iter_debug==0:
                rgb       = rgbs[idx]

                img_gt     = draw_rpn_gt(rgb, batch_gt_boxes2d, batch_gt_labels)
                rgb_label  = draw_rpn_labels (img_gt, anchors_rgb, batch_rgb_inds, batch_rgb_labels )
                rgb_target = draw_rpn_targets(rgb, anchors_rgb, batch_rgb_pos_inds, batch_rgb_targets)
                #imshow('img_rpn_gt',img_gt)
                imshow('img_rgb_label',rgb_label)
                imshow('img_rpn_target',rgb_target)

                img_label  = draw_rcnn_labels (rgb, batch_rgb_rois, batch_fuse_labels )
                img_target = draw_rcnn_targets(rgb, batch_rgb_rois, batch_fuse_labels, batch_fuse_targets2d)
                imshow('img_rcnn_label',img_label)
                imshow('img_rcnn_target',img_target)

                img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
                imshow('img_rgb_rois',img_rgb_rois)

                projections=box_transform_3dTo2D_inv(batch_rgb_rois[:,1:],batch_fuse_targets_3dTo2Ds)
                img_rcnn_3dTo2D = draw_rgb_projections(rgb, projections, color=(0,0,255), thickness=1)
                imshow('img_rcnn_3dTo2D',img_rcnn_3dTo2D)
                # plt.pause(0.5)
                cv2.waitKey(500)

            ## run classification and regression loss -----------
            fd2={
                **fd1,

                rgb_images: batch_rgb_images,
                rgb_rois:   batch_rgb_rois,

                rgb_inds:     batch_rgb_inds,
                rgb_pos_inds: batch_rgb_pos_inds,
                rgb_labels:   batch_rgb_labels,
                rgb_targets:  batch_rgb_targets,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets2d,

                fuse_targets_3dTo2Ds: batch_fuse_targets_3dTo2Ds
            }

            _, rcnn_probs, batch_rgb_cls_loss, batch_rgb_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss, batch_fuse_reg_loss_dTo2D = \
               sess.run([solver_step, fuse_probs, rgb_cls_loss, rgb_reg_loss, fuse_cls_loss, fuse_reg_loss, fuse_reg_loss_3dTo2D],fd2)

            speed=time.time()-start_time
            log.write('%5.1f   %5d    %0.4fs   %0.6f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  |%0.5f   \n' %\
                (epoch, iter, speed, rate, batch_rgb_cls_loss, batch_rgb_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss, batch_fuse_reg_loss_dTo2D))
            
            # debug: ------------------------------------

            # if vis and iter%iter_debug==0:
            #     top_image = top_imgs[idx]
            #     rgb       = rgbs[idx]

            #     batch_top_probs, batch_top_scores, batch_top_deltas  = \
            #         sess.run([ top_probs, top_scores, top_deltas ],fd2)

            #     batch_fuse_probs, batch_fuse_deltas, batch_fuse_deltas_2d = \
            #         sess.run([ fuse_probs, fuse_deltas, fuse_deltas_2d ],fd2)
            #     # pdb.set_trace()
            #     #batch_fuse_deltas=0*batch_fuse_deltas #disable 3d box prediction
            #     probs, boxes3d, boxes2d = rcnn_nms_2d(batch_fuse_probs, batch_fuse_deltas, batch_rois3d_old, batch_fuse_deltas_2d, batch_rgb_rois[:,1:],rgb_shape, threshold=0.05)

            #     ## show rpn(top) nms
            #     img_rpn     = draw_rpn    (top_image, batch_top_probs, batch_top_deltas, anchors, inside_inds)
            #     # img_rpn_nms = draw_rpn_nms(top_image, batch_proposals, batch_proposal_scores)
            #     #imshow('img_rpn',img_rpn)
            #     if vis :
            #         # imshow('img_rpn_nms',img_rpn_nms)
            #         cv2.waitKey(1)

            #     ## show rcnn(fuse) nms
            #     img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
            #     rgb1 = draw_rcnn_nms(rgb, boxes3d, probs)
            #     projections=box3d_to_rgb_projections(gen_rois3D)
            #     img_rcnn_nms = draw_rgb_projections(rgb1, projections, color=(0,0,255), thickness=1)
            #     if vis :
            #         imshow('img_rcnn',img_rcnn)
            #         imshow('img_rcnn_nms',img_rcnn_nms)
            #         cv2.waitKey(0)
            if (iter)%10==0:
                summary = sess.run(merged,fd2)
                train_writer.add_summary(summary, iter)
            # save: ------------------------------------
            
            if (iter)%5000==0 and (iter!=0):
                saver.save(sess, out_dir + '/check_points/snap_2dTo3d_with_2d_pretrained_traintxt_%06d.ckpt'%iter)  #iter
                # saver_rgb.save(sess, out_dir + '/check_points/pretrained_Res_rgb_model%06d.ckpt'%iter)
                # saver_top.save(sess, out_dir + '/check_points/pretrained_Res_top_model%06d.ckpt'%iter)
                # pdb.set_trace()

            idx=idx+1






## main function ##########################################################################

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()