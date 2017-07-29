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
import glob
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
label_path = os.path.join(train_data_root, "detection_labels")
img_path = os.path.join(train_data_root, "Raw_Images")
# calib_path = os.path.join(kitti_dir, "calib/")
# train_data_root='/home/hhs/4T/datasets/dummy_datas/seg'
classes = {'__background__':0, 'car':1}#, ' Van':1, 'Truck':1, 'Tram':1}
# result_path='./evaluate_object/val_gt/'
gt_boxes2d_path = train_data_root + '/gt_boxes2d'
gt_labels_path = train_data_root + '/gt_labels'


empty(gt_boxes2d_path)
empty(gt_labels_path)

makedirs(gt_boxes2d_path)
makedirs(gt_labels_path)


img_list=glob.glob(label_path + '/*.txt')
image_inds=[imflie.strip().split('/')[-1].split('.')[0] for imflie in img_list]
num=len(img_list)
for i in range(num):

	filename = os.path.join(label_path, image_inds[i] + ".txt")
	Imgname = os.path.join(img_path, image_inds[i] + ".png")
	image = cv2.imread(Imgname)
	print("Processing: ", filename)
	with open(filename, 'r') as f:
		lines = f.readlines()

	num_objs = len(lines)
	if num_objs == 0:
		continue

	gt_boxes2d = []
	gt_labels  = []

	for j in range(num_objs):
		obj=lines[j].strip().split(' ')
		try:
			clss=classes[obj[0].strip()]
			# file=open(result_path+'%06d'%i+'.txt', 'a')
			
		except:
			continue
		

		x1 = float(obj[1])
		y1 = float(obj[2])
		x2 = float(obj[3])
		y2 = float(obj[4])

		box2d=np.array([x1, y1, x2, y2])

		gt_boxes2d.append(box2d)
		gt_labels.append(clss)

	# img_2d_gt = draw_boxes(image, gt_boxes2d, color=(255,0,255), thickness=1)
	# imshow('img_2d_gt',img_2d_gt)
	# cv2.waitKey(20)

	gt_boxes2d = np.array(gt_boxes2d,dtype=np.float32)
	gt_labels  = np.array(gt_labels ,dtype=np.uint8)

	
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


	np.save(gt_boxes2d_path+'/gt_boxes2d_%s.npy'%image_inds[i],gt_boxes2d)
	np.save(gt_labels_path+'/gt_labels_%s.npy'%image_inds[i],gt_labels)



#Generate train and val list
#3DOP train val list http://www.cs.toronto.edu/objprop3d/data/ImageSets.tar.gz
files_list=glob.glob(gt_labels_path+"/gt_labels_*.npy")
index=np.array([file_index.strip().split('_')[-1].split('.')[0] for file_index in files_list ])
num_frames=len(files_list)
train_num=int(np.round(num_frames*0.7))
random_index=np.random.permutation(index)
train_list=random_index[:train_num]
val_list=random_index[train_num:]
np.save(train_data_root+'/train_list.npy',train_list)
np.save(train_data_root+'/val_list.npy',val_list)
np.save(train_data_root+'/train_val_list.npy',random_index)






