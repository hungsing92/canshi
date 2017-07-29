from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from net.processing.cpu_nms import cpu_nms as nms


def draw_rcnn_berfore_nms(image, probs,  deltas, rois, rois3d, threshold=0.8):

    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>threshold)[0]

    #post processing
    rois   = rois[idx]
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]

    num = len(rois)
    for n in range(num):
        a   = rois[n,1:5]
        cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(rois[:,1:5],deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        draw_box3d_on_top(image,boxes3d)


#after nms : camera image
def draw_rcnn_nms_rgb(rgb, boxes3d, probs, darker=0.7):

    img_rcnn_nms = rgb.copy()*darker
    projections = box3d_to_rgb_projections(boxes3d)
    img_rcnn_nms = draw_rgb_projections(img_rcnn_nms,  projections, color=(255,255,255), thickness=1)

    return img_rcnn_nms

#after nms : lidar top image
def draw_rcnn_after_nms_top(image, boxes3d, probs):
    draw_box3d_on_top(image,boxes3d)



def draw_rcnn(image, probs,  deltas, rois, rois3d, threshold=0.1, darker=0.7):

    img_rcnn = image.copy()*darker
    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>threshold)[0]

    #post processing
    rois   = rois[idx]
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]

    num = len(rois)
    for n in range(num):
        a   = rois[n,1:5]
        cv2.rectangle(img_rcnn,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(rois[:,1:5],deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        img_rcnn = draw_box3d_on_top(img_rcnn,boxes3d)

    return img_rcnn



def draw_rcnn_nms(rgb, boxes3d, probs, darker=0.7):

    img_rcnn_nms = rgb.copy()*darker
    projections = box3d_to_rgb_projections(boxes3d)
    img_rcnn_nms = draw_rgb_projections(img_rcnn_nms,  projections, color=(255,255,255), thickness=1)

    return img_rcnn_nms




## temporay post-processing ....
## <todo> to be updated


def rcnn_nms( probs,  deltas,  rois3d,  threshold = 0.05):


    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>0.5)[0]

    #post processing
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]
    probs  = probs [idx]

    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(priors,deltas)
        return probs,boxes


    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        top_boxes=box3d_to_top_box(boxes3d)
        keep = nms(np.hstack((top_boxes, probs.reshape(-1,1))), threshold)
        boxes3d=boxes3d[keep]
        boxes3d  = regularise_box3d(boxes3d)

        return probs, boxes3d

def IoM(rect_1, rect_2):
    '''
    :param rect_1: list in format [x11, y11, x12, y12, confidence]
    :param rect_2:  list in format [x21, y21, x22, y22, confidence]
    :return:    returns IoM ratio (intersection over min-area) of two rectangles
    '''
    zeros_=np.zeros((len(rect_1), 1), dtype=np.float32)
    if len(rect_2) == 1:
        rect_2 = np.tile(rect_2,(len(rect_1),1))
    x11 = rect_1[:,0].reshape(-1,1)    # first rectangle top left x
    y11 = rect_1[:,1].reshape(-1,1)    # first rectangle top left y
    x12 = rect_1[:,2].reshape(-1,1)    # first rectangle bottom right x
    y12 = rect_1[:,3].reshape(-1,1)    # first rectangle bottom right y
    x21 = rect_2[:,0].reshape(-1,1)    # second rectangle top left x
    y21 = rect_2[:,1].reshape(-1,1)    # second rectangle top left y
    x22 = rect_2[:,2].reshape(-1,1)    # second rectangle bottom right x
    y22 = rect_2[:,3].reshape(-1,1)    # second rectangle bottom right y
    x_overlap = np.max([zeros_, np.min([x12,x22],axis=0) -np.max([x11,x21],axis=0)],axis=0)
    y_overlap = np.max([zeros_, np.min([y12,y22],axis=0) -np.max([y11,y21],axis=0)],axis=0)
    intersection = x_overlap * y_overlap
    rect1_area = (y12 - y11) * (x12 - x11)
    rect2_area = (y22 - y21) * (x22 - x21)
    min_area = np.min([rect1_area, rect2_area],axis=0)
    iom = intersection.reshape(-1,1) / min_area.reshape(-1,1)
    area_compare = np.where(rect1_area<10*rect2_area)[0]

    return iom, area_compare

def rcnn_nms_2d( probs,  deltas,  rois2d, threshold = 0.05):


    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>0.8)[0]

    #post processing

    rois2d = rois2d[idx]
    deltas = deltas[idx,cls].reshape(-1,4)
    probs  = probs [idx]

    boxes2d = box2d_transform_inv(rois2d[:,1:], deltas)
    keep = nms(np.hstack((boxes2d, probs.reshape(-1,1))), threshold)

    probs    = probs[keep]
    boxes2d = boxes2d[keep]

    return probs, boxes2d




# def rcnn_nms_2d( probs,  deltas,  rois3d, deltas2d, rois2d, rgb_shape, threshold = 0.05):


#     cls=1  # do for class-one only
#     probs = probs[:,cls] #see only class-1
#     idx = np.where(probs>0.6)[0]

#     #post processing
#     rois3d = rois3d[idx]
#     rois2d = rois2d[idx]
#     deltas = deltas[idx,cls]
#     deltas2d = deltas2d[idx,cls]
#     probs  = probs [idx]

#     if deltas.shape[1:]==(4,):
#         boxes = box_transform_inv(priors,deltas)
#         return probs,boxes


#     if deltas.shape[1:]==(8,3):
#         boxes3d  = box3d_transform_inv(rois3d, deltas)
#         top_boxes=box3d_to_top_box(boxes3d)
#         keep = nms(np.hstack((top_boxes, probs.reshape(-1,1))), threshold)
#         boxes3d=boxes3d[keep]
#         boxes3d  = regularise_box3d(boxes3d)
#         probs    = probs[keep]

#         boxes2d = box2d_transform_inv(rois2d, deltas2d)
#         boxes2d = boxes2d[keep]

#         rgb_boxes=project_to_rgb(boxes3d, rgb_shape[1], rgb_shape[0] )
#         image = np.array([0, 0, rgb_shape[1], rgb_shape[0]]).reshape(-1,4)
#         iom, area_compare = IoM(rgb_boxes, image)

#         keep = np.intersect1d(np.where(iom>0.1)[0],area_compare)

#         boxes3d = boxes3d[keep]
#         boxes2d = boxes2d[keep]
#         probs    = probs[keep]

#         return probs, boxes3d, boxes2d