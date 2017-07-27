import _init_paths
from net.common import *
from net.processing.boxes3d import *
from net.utility.draw import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from time import time
from net.utility.file import *
import cv2
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================

def project_velo2rgb(velo,Tr,R0,P2):
	T=np.zeros([4,4],dtype=np.float32)
	T[:3,:]=Tr
	T[3,3]=1
	R=np.zeros([4,4],dtype=np.float32)
	R[:3,:3]=R0
	R[3,3]=1
	num=len(velo)
	projections = np.zeros((num,8,2),  dtype=np.int32)
	for i in range(len(velo)):
		box3d=np.ones([8,4],dtype=np.float32)
		box3d[:,:3]=velo[i]
		M=np.dot(P2,R)
		M=np.dot(M,T)
		box2d=np.dot(M,box3d.T)
		box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
		projections[i] = box2d
	return projections

def load_kitti_calib(calib_path,index):
    """
    load projection matrix
    """
    calib_dir = os.path.join(calib_path, str(index).zfill(6) + '.txt')

    P0 = np.zeros(12, dtype=np.float32)
    P1 = np.zeros(12, dtype=np.float32)
    P2 = np.zeros(12, dtype=np.float32)
    P3 = np.zeros(12, dtype=np.float32)
    R0 = np.zeros(9, dtype=np.float32)
    Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
    Tr_imu_to_velo = np.zeros(12, dtype=np.float32)
    with open(calib_dir) as fi:
        lines = fi.readlines()
        assert(len(lines) == 8)
    
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo= np.array(obj, dtype=np.float32)
        
    return {'P2' : P2.reshape(3,4),
            'R0' : R0.reshape(3,3),
            'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}



# kitti_dir = "/home/hhs/4T/datasets/KITTI/object/training"
label_path = os.path.join(kitti_dir, "label_2/")
img_path = os.path.join(kitti_dir, "image_2/")
calib_path = os.path.join(kitti_dir, "calib/")
# train_data_root='/home/hhs/4T/datasets/dummy_datas/seg'
classes = {'__background__':0, 'Car':1, 'Van':1}#, ' Van':1, 'Truck':1, 'Tram':1}
# result_path='./evaluate_object/val_gt/'
gt_boxes3d_path = train_data_root + '/gt_boxes3d'
gt_boxes2d_path = train_data_root + '/gt_boxes2d'
gt_labels_path = train_data_root + '/gt_labels'
gt_3dTo2D_path = train_data_root + '/gt_3dTo2D'

empty(gt_boxes3d_path)
empty(gt_boxes2d_path)
empty(gt_labels_path)
empty(gt_3dTo2D_path)

makedirs(gt_boxes3d_path)
makedirs(gt_boxes2d_path)
makedirs(gt_labels_path)
makedirs(gt_3dTo2D_path)

width = []
length = []
ratio = []
for i in range(7481):

	calib=load_kitti_calib(calib_path,i)
	Tr = calib['Tr_velo2cam']
	P2 = calib['P2']
	R0 = calib['R0']
	filename = os.path.join(label_path, str(i).zfill(6) + ".txt")
	Imgname = os.path.join(img_path, str(i).zfill(6) + ".png")
	image = cv2.imread(Imgname)
	print("Processing: ", filename)
	with open(filename, 'r') as f:
		lines = f.readlines()

	num_objs = len(lines)
	if num_objs == 0:
		continue
	gt_boxes3d = []
	gt_boxes2d = []
	gt_labels  = []
	cam=np.ones([4,1])

	for j in range(num_objs):
		obj=lines[j].strip().split(' ')
		try:
			clss=classes[obj[0].strip()]
			# file=open(result_path+'%06d'%i+'.txt', 'a')
			
		except:
			continue
		
		truncated = float(obj[1])
		occluded = float(obj[2])
		x1 = float(obj[4])
		y1 = float(obj[5])
		x2 = float(obj[6])
		y2 = float(obj[7])
		h = float(obj[8])
		w = float(obj[9])
		l = float(obj[10])
		tx = float(obj[11])
		ty = float(obj[12])
		tz = float(obj[13])
		ry = float(obj[14])

		width.append(w)
		length.append(l)
		ratio.append(w/l)

		cam[0]=tx
		cam[1]=ty
		cam[2]=tz
		t_lidar=project_cam2velo(cam,Tr)
		Box = np.array([ # in velodyne coordinates around zero point and without orientation yet\
	        [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
	        [ l/2, l/2,  -l/2, -l/2, l/2, l/2,  -l/2, -l/2], \
	        [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
		rotMat = np.array([\
	          [np.cos(ry), +np.sin(ry), 0.0], \
	          [-np.sin(ry),  np.cos(ry), 0.0], \
	          [        0.0,          0.0, 1.0]])

		cornerPosInVelo = np.dot(rotMat, Box) + np.tile(t_lidar, (8,1)).T
		box3d=cornerPosInVelo.transpose()
		box2d=np.array([x1, y1, x2, y2])

		rgb_boxes=project_to_rgb_roi([box3d], image.shape[1], image.shape[0])
		# line='Car %.2f %d -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n'%(truncated, occluded, rgb_boxes[0][1], rgb_boxes[0][2], rgb_boxes[0][3], rgb_boxes[0][4], 1)
		# file.write(line)
		
		top_box=box3d_to_top_box([box3d])
		if (top_box[0][0]>=Top_X0) and (top_box[0][1]>=Top_Y0) and (top_box[0][2]<=Top_Xn) and (top_box[0][3]<=Top_Yn):
			gt_boxes3d.append(box3d)
			gt_boxes2d.append(box2d)
			gt_labels.append(clss)

	gt_3dTo2D=project_velo2rgb(gt_boxes3d,Tr,R0,P2)

	# img_rcnn_nms = draw_rgb_projections(image, rgb_projections, color=(0,0,255), thickness=1)
	# imshow('draw_rcnn_nms',img_rcnn_nms)
	# cv2.waitKey(0)


	# pdb.set_trace()
	# if len(gt_labels) == 0:
	# 	continue
	# file.close()
	gt_boxes3d = np.array(gt_boxes3d,dtype=np.float32)
	gt_boxes2d = np.array(gt_boxes2d,dtype=np.float32)
	gt_labels  = np.array(gt_labels ,dtype=np.uint8)
	gt_3dTo2D = np.array(gt_3dTo2D)
	
# plt.hist(width,50,normed=1,facecolor='g',alpha=0.75)
# plt.grid(True)
# plt.show()
# pdb.set_trace()
# plt.hist(length,50,normed=1,facecolor='g',alpha=0.75)
# plt.grid(True)
# plt.show()
# pdb.set_trace()
# plt.hist(ratio,50,normed=1,facecolor='g',alpha=0.75)
# plt.grid(True)
# plt.show()
# pdb.set_trace()

# from mpl_toolkits.mplot3d import Axes3D
# ax=plt.subplot(111,projection='3d')
# hist, xedges, yedges = np.histogram2d(width, length, bins=50)
# elements = (len(xedges) - 1) * (len(yedges) - 1)
# xpos, ypos = np.meshgrid(xedges[:-1]+.25, yedges[:-1]+.25)
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros(elements)
# dx = .1 * np.ones_like(zpos)
# dy = dx.copy()
# dz = hist.flatten()
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.4)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.show()

	np.save(gt_boxes3d_path+'/gt_boxes3d_%05d.npy'%i,gt_boxes3d)
	np.save(gt_boxes2d_path+'/gt_boxes2d_%05d.npy'%i,gt_boxes2d)
	np.save(gt_labels_path+'/gt_labels_%05d.npy'%i,gt_labels)
	np.save(gt_3dTo2D_path+'/gt_3dTo2D_%05d.npy'%i,gt_3dTo2D)


# #Generate train and val list
# #3DOP train val list http://www.cs.toronto.edu/objprop3d/data/ImageSets.tar.gz
# files_list=glob.glob(gt_labels_path+"/gt_labels_*.npy")
# index=np.array([file_index.strip().split('_')[-1].split('.')[0] for file_index in files_list ])
# num_frames=len(files_list)
# train_num=int(np.round(num_frames*0.7))
# random_index=np.random.permutation(index)
# train_list=random_index[:train_num]
# val_list=random_index[train_num:]
# np.save(train_data_root+'/train_list.npy',train_list)
# np.save(train_data_root+'/val_list.npy',val_list)
# np.save(train_data_root+'/train_val_list.npy',random_index)






