from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *
import pdb


# gt_boxes    : (x1,y1,  x2,y2  label)  #projected 2d
# gt_boxes_3d : (x1,y1,z1,  x2,y2,z2,  ....    x8,y8,z8,  label)


def rcnn_target(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets

def _add_jittered_boxes(rois, batch_inds, gt_boxes, jitter=0.1):
    ws = gt_boxes[:, 2] - gt_boxes[:, 0]
    hs = gt_boxes[:, 3] - gt_boxes[:, 1]
    shape = len(gt_boxes)
    jitter = np.random.uniform(-jitter, jitter, [shape, 1])
    jitter = jitter.reshape(-1,1)
    ws_offset = ws * jitter
    hs_offset = hs * jitter
    x1s = gt_boxes[:, 0] + ws_offset
    x2s = gt_boxes[:, 2] + ws_offset
    y1s = gt_boxes[:, 1] + hs_offset
    y2s = gt_boxes[:, 3] + hs_offset
    boxes = np.hstack([
                    x1s[:, tf.newaxis],
                y1s[:, tf.newaxis],
                x2s[:, tf.newaxis],
                y2s[:, tf.newaxis]])
    new_batch_inds = np.zeros([shape], np.int32)
    zeros         = np.zeros((shape, 1), dtype=np.float32)
    return np.vstack([rois,np.hstack((zeros, boxes))]) \

def _add_jittered_boxes(gt_boxes, top_box_thresh = 0.15, nums_of_augment = 5):
    gt_nums = len(gt_boxes)
    top_rois = np.zeros((gt_nums*nums_of_augment,5))
    inds = 0
    # np.random.seed(None)
    for i in range(gt_nums):
        top_box = gt_boxes[i]
        width = top_box[2]- top_box[0]
        height = top_box[3]- top_box[1]
        for j in range(nums_of_augment):
            x1 = top_box[0]+np.random.uniform(-top_box_thresh,top_box_thresh)*width
            x2 = top_box[2]+np.random.uniform(-top_box_thresh,top_box_thresh)*width
            y1 = top_box[1]+np.random.uniform(-top_box_thresh,top_box_thresh)*height
            y2 = top_box[3]+np.random.uniform(-top_box_thresh,top_box_thresh)*height
            top_rois[inds,1:] = np.array([x1,y1,x2,y2])#,dtype=np.int32)

            inds = inds+1
    return top_rois

def rcnn_target_3dTo2D(rois, gt_labels, gt_boxes, gt_3dTo2Ds, width, height):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    jittered_rois = _add_jittered_boxes(gt_boxes ,0.1, 3)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, jittered_rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'

    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    # print('rois_per_image: ',rois_per_image)
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0

    gt_boxes2d = gt_boxes[gt_assignment[keep]]
    gt_3dTo2D_ = gt_3dTo2Ds[gt_assignment[keep]]
    # pdb.set_trace()
    gt_3dTo2D_ = gt_3dTo2D_.reshape(-1,16)

    et_boxes=rois[:,1:5]

    targets_2d = box_transform_2d(rois, gt_boxes2d)
    targets_3dTo2Ds = box_transform_3dTo2D(et_boxes, gt_3dTo2D_)
    # targets_3dTo2Ds = box_transform_3dTo2D_new_loss(et_boxes, gt_3dTo2D_)

        #exit(0)
    return rois, labels, targets_2d, targets_3dTo2Ds



def draw_rcnn_labels(image, rois,  labels, darker=0.7):
    is_print=0

    ## draw +ve/-ve labels ......
    boxes = rois[:,1:5].astype(np.int32)
    labels = labels.reshape(-1)
    # pdb.set_trace()
    fg_label_inds = np.where(labels != 0)[0]
    bg_label_inds = np.where(labels == 0)[0]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rcnn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    img_label = image.copy()*darker
    if 1:
        for i in bg_label_inds:
            a = boxes[i]
            cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (32,32,0), 1)
            cv2.circle(img_label,(a[0], a[1]),2, (32,32,0), -1)

    for i in fg_label_inds:
        a = boxes[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (255,0,255), -1)

    return img_label

def draw_rcnn_targets(image, rois, labels,  targets, darker=0.7):
    is_print=1

    #draw +ve targets ......
    boxes = rois[:,1:5].astype(np.int32)

    fg_target_inds = np.where(labels != 0)[0]
    num_pos_target = len(fg_target_inds)
    if is_print: print ('rcnn target : num_pos=%d'  %(num_pos_target))

    img_target = image.copy()*darker
    for n,i in enumerate(fg_target_inds):
        a = boxes[i]
        cv2.rectangle(img_target,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)

        if targets.shape[1:]==(4,):
            t = targets[n]
            # b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
            b = box2d_transform_inv(a.reshape(1,4), t.reshape(1,4))
            b = b.reshape(4).astype(np.int32)
            cv2.rectangle(img_target,(b[0], b[1]), (b[2], b[3]), (255,255,255), 1)

        if targets.shape[1:]==(8,3):
            t = targets[n]
            a3d = top_box_to_box3d(a.reshape(1,4))
            b3d = box3d_transform_inv(a3d, t.reshape(1,8,3)).astype(np.int32)
            #b3d = b3d.reshape(1,8,3)
            img_target = draw_box3d_on_top(img_target, b3d, darken=1)

    return img_target





