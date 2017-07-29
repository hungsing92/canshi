from net.common import *


##extension for 3d
def generate_3d_boxes_samples(gt_top_boxes,gt_boxesZ):
    gt_nums = len(gt_top_boxes)
    nums_of_augment = 5
    top_box_thresh = 0.3
    proposals_z_thresh = 0.15
    top_rois = np.zeros((gt_nums*nums_of_augment,5))
    proposals_z = np.zeros((gt_nums*nums_of_augment,2))
    inds = 0
    # np.random.seed(None)
    for i in range(gt_nums):
        top_box = gt_top_boxes[i]
        z0, zn= gt_boxesZ[i]
        z_height = zn-z0
        width = top_box[2]- top_box[0]
        height = top_box[3]- top_box[1]
        for j in range(nums_of_augment):
            x1 = top_box[0]+np.random.uniform(-top_box_thresh,top_box_thresh)*width
            x2 = top_box[2]+np.random.uniform(-top_box_thresh,top_box_thresh)*width
            y1 = top_box[1]+np.random.uniform(-top_box_thresh,top_box_thresh)*height
            y2 = top_box[3]+np.random.uniform(-top_box_thresh,top_box_thresh)*height
            z0_ = z0 +np.random.uniform(-proposals_z_thresh,proposals_z_thresh)*z_height
            zn_ = zn +np.random.uniform(-proposals_z_thresh,proposals_z_thresh)*z_height
            top_rois[inds,1:] = np.array([x1,y1,x2,y2])#,dtype=np.int32)
            proposals_z[inds,:] = np.array([z0_,zn_])
            inds = inds+1
    return top_rois, proposals_z

def get_boxes3d_z(boxes3d):
    boxes3d_z=np.zeros((len(boxes3d),2), dtype=np.float32)
    for num in np.arange(len(boxes3d)):
        box3d = boxes3d[num]
        # center = np.sum(box3d,axis=0, keepdims=True)/8
        # dis=0
        # for k in range(0,4):
        #     i,j=k,k+4
        #     dis +=np.sum((box3d[i]-box3d[j])**2) **0.5
        # h = dis/4
        # z0 = center[:,2]-h/2
        # zn = center[:,2]+h/2
        z0 = np.min(box3d[:,2])
        zn = np.max(box3d[:,2])
        boxes3d_z[num,:]=np.array([z0,zn]).reshape(-1,2)
    return boxes3d_z

def project_cam2velo(cam,Tr):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=Tr
    T[3,3]=1
    T_inv=np.linalg.inv(T)
    lidar_loc_=np.dot(T_inv,cam)
    lidar_loc=lidar_loc_[:3]
    return lidar_loc.reshape(1,3)

def project_velo2cam(velo):
    lidar_loc_=np.dot(MATRIX_TR,velo)
    lidar_loc=lidar_loc_[:3]
    return lidar_loc[0],lidar_loc[1],lidar_loc[2]
    

def project_to_roi3d(top_rois):
    num = len(top_rois)
    rois3d = np.zeros((num,8,3))
    rois3d = top_box_to_box3d(top_rois[:,1:5])
    return rois3d


def project_to_rgb_roi(rois3d, width, height):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    projections = box3d_to_rgb_projections(rois3d)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:,0])
        maxx = np.max(qs[:,0])
        miny = np.min(qs[:,1])
        maxy = np.max(qs[:,1])
        minx = np.maximum(np.minimum(minx, width - 1), 0)
        maxx = np.maximum(np.minimum(maxx, width - 1), 0)
        miny = np.maximum(np.minimum(miny, height - 1), 0)
        maxy = np.maximum(np.minimum(maxy, height - 1), 0)
        rois[n,1:5] = minx,miny,maxx,maxy

    return rois


def project_to_rgb(rois3d, width, height):
    num  = len(rois3d)
    rois = np.zeros((num,4),dtype=np.int32)
    projections = box3d_to_rgb_projections(rois3d)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:,0])
        maxx = np.max(qs[:,0])
        miny = np.min(qs[:,1])
        maxy = np.max(qs[:,1])
        rois[n,:] = minx,miny,maxx,maxy

    return rois


def  project_to_front_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)

    return rois



def top_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    y = Yn*TOP_Y_DIVISION-(xx+0.5)*TOP_Y_DIVISION + TOP_Y_MIN
    x = Xn*TOP_X_DIVISION-(yy+0.5)*TOP_X_DIVISION + TOP_X_MIN

    return x,y


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    xx = Yn-int((y-TOP_Y_MIN)//TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_DIVISION)

    return xx,yy


def top_box_to_box3d(boxes):

    num=len(boxes)
    boxes3d = np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        x1,y1,x2,y2 = boxes[n]
        points = [ (x1,y2), (x2,y2), (x2,y1), (x1,y1) ]
        for k in range(4):
            xx,yy = points[k]
            x,y  = top_to_lidar_coords(xx,yy)
            boxes3d[n,k,  :] = x,y, -1.9  ## <todo>-2
            boxes3d[n,4+k,:] = x,y,0.2  #0.4

    return boxes3d

def top_z_to_box3d(boxes,proposals_z):

    num=len(boxes)
    boxes3d = np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        x1,y1,x2,y2 = boxes[n]
        z0,zn = proposals_z[n]

        points = [ (x1,y2), (x2,y2), (x2,y1), (x1,y1) ]
        for k in range(4):
            xx,yy = points[k]
            x,y  = top_to_lidar_coords(xx,yy)
            boxes3d[n,k,  :] = x,y, z0  ## <todo>-2
            boxes3d[n,4+k,:] = x,y, zn  #0.4

    return boxes3d



def box3d_to_top_box(boxes3d):

    num  = len(boxes3d)
    boxes = np.zeros((num,4),  dtype=np.float32)

    for n in range(num):
        b   = boxes3d[n]

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)

        boxes[n]=np.array([umin,vmin,umax,vmax])

    return boxes



def box3d_to_rgb_projections(boxes3d, Mt=None, Kt=None):

    if Mt is None: Mt = np.array(MATRIX_Mt)
    if Kt is None: Kt = np.array(MATRIX_Kt)

    num  = len(boxes3d)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for n in range(num):
        box3d = boxes3d[n]
        Ps = np.hstack(( box3d, np.ones((8,1))) )
        Qs = np.matmul(Ps,Mt)
        Qs = Qs[:,0:3]
        qs = np.matmul(Qs,Kt)
        zs = qs[:,2].reshape(8,1)
        qs = (qs/zs)
        projections[n] = qs[:,0:2]

    return projections


def box3d_to_top_projections(boxes3d):

    num = len(boxes3d)
    projections = np.zeros((num,4,2),  dtype=np.float32)
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)
        projections[n] = np.array([[u0,v0],[u1,v1],[u2,v2],[u3,v3]])

    return projections


def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=0.7):
    # def length_filter(x1,x2,img):
    #     x1=np.array(x1)
    #     x2=np.array(x2)
    #     dist=np.sqrt(np.sum((x1-x2)**2))
    #     if dist>0.75*img.shape[1]:
    #         return True
    #     else:
    #         return False

    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        # cv2.putText(image,"%d"%n, (qs[6,0],qs[6,1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            # if length_filter((qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),img):
            #     break
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)

    return img


def draw_box3d_on_top(image, boxes3d,color=(255,255,255), thickness=1, darken=0.7):

    img = image.copy()*darken
    num =len(boxes3d)
    for n in range(num):
        b   = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)
        cv2.line(img, (u0,v0), (u1,v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1,v1), (u2,v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2,v2), (u3,v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3,v3), (u0,v0), color, thickness, cv2.LINE_AA)

    return  img

def draw_boxes(image, boxes, color=(0,0,255), thickness=1, darken=0.7):

    img = image.copy()#*darken
    num =len(boxes)
    for n in range(num):
        b = boxes[n]
        cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),color,thickness)

    return img


## regression -------------------------------------------------------
##<todo> refine this normalisation later ... e.g. use log(scale)
def box3d_transform0(et_boxes3d, gt_boxes3d):

    et_centers =   np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =   10#*np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    deltas = (et_boxes3d-gt_boxes3d)/et_scales
    return deltas


def box3d_transform_inv0(et_boxes3d, deltas):

    et_centers =  np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =  10#*np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    boxes3d = -deltas*et_scales+et_boxes3d

    return boxes3d

# rotMat = np.array([\
#               [np.cos(np.pi/2), +np.sin(np.pi/2), 0.0], \
#               [-np.sin(np.pi/2),  np.cos(np.pi/2), 0.0], \
#               [        0.0,          0.0, 1.0]])

def box3d_transform(et_boxes3d, gt_boxes3d):

    num=len(et_boxes3d)
    deltas=np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        e=et_boxes3d[n]
        center = np.sum(e,axis=0, keepdims=True)/8
        scale = (np.sum((e-center)**2)/8)**0.5

        g=gt_boxes3d[n]
        deltas[n]= (g-e)/scale
    return deltas

def box_transform_2d(et_boxes2d, gt_boxes):
    # pdb.set_trace()
    et_boxes=et_boxes2d[:,1:]
    num=len(et_boxes)
    deltas=np.zeros((num,4),dtype=np.float32)
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    scale = ((0.5*et_ws)**2+(0.5*et_hs)**2)**0.5
    deltas = (gt_boxes-et_boxes)/scale.reshape(-1,1)
    return deltas

def box2d_transform_inv(et_boxes, deltas):
    num=len(et_boxes)
    boxes2d=np.zeros((num,4),dtype=np.float32)    
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    scale = ((0.5*et_ws)**2+(0.5*et_hs)**2)**0.5
    boxes2d = deltas*scale.reshape(-1,1)+et_boxes
    return boxes2d

def box_transform_3dTo2D(et_boxes, gt_3dTo2D):
    num=len(et_boxes)
    deltas=np.zeros((num,16),dtype=np.float32)
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    c_xs   = (et_boxes[:, 2] + et_boxes[:, 0])/2
    c_ys   = (et_boxes[:, 3] + et_boxes[:, 1])/2
    center= np.tile(np.hstack([c_xs.reshape(-1,1),c_ys.reshape(-1,1)]),(1,8))
    scale = ((0.5*et_ws)**2+(0.5*et_hs)**2)**0.5
    deltas = (gt_3dTo2D-center)/scale.reshape(-1,1)
    return deltas

def box_transform_3dTo2D_inv(et_boxes,targets_3dTo2Ds):
    num=len(et_boxes)
    points_3dTo2D=np.zeros((num,16),dtype=np.int32) 
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    c_xs   = (et_boxes[:, 2] + et_boxes[:, 0])/2
    c_ys   = (et_boxes[:, 3] + et_boxes[:, 1])/2
    center= np.tile(np.hstack([c_xs.reshape(-1,1),c_ys.reshape(-1,1)]),(1,8))
    scale = ((0.5*et_ws)**2+(0.5*et_hs)**2)**0.5
    points_3dTo2D=targets_3dTo2Ds*scale.reshape(-1,1)+center
    return points_3dTo2D.reshape(-1,8,2).astype(np.int32)

def box3d_transform_inv(et_boxes3d, deltas):

    num=len(et_boxes3d)
    boxes3d=np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        e=et_boxes3d[n]
        center = np.sum(e,axis=0, keepdims=True)/8
        scale = (np.sum((e-center)**2)/8)**0.5

        d=deltas[n]
        boxes3d[n]= e+scale*d

    return boxes3d


##<todo> refine this regularisation later
def regularise_box3d(boxes3d):

    num = len(boxes3d)
    reg_boxes3d =np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        b=boxes3d[n]

        dis=0
        corners = np.zeros((4,3),dtype=np.float32)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,k+4
            dis +=np.sum((b[i]-b[j])**2) **0.5
            corners[k] = (b[i]+b[j])/2

        dis = dis/4
        b = reg_boxes3d[n]
        for k in range(0,4):
            i,j=k,k+4
            b[i]=corners[k]-dis/2*np.array([0,0,1])
            b[j]=corners[k]+dis/2*np.array([0,0,1])

    return reg_boxes3d