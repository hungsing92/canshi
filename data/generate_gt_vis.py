import _init_paths
from net.common import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import time
import glob
import cv2
from net.utility.draw import *
import mayavi.mlab as mlab
from data import *
from net.utility.file import *
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================

## lidar to top ##
def lidar_to_top(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2
    v16=[]

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    filter_x=np.where((pxs>=TOP_X_MIN) & (pxs<=TOP_X_MAX))[0]
    filter_y=np.where((pys>=TOP_Y_MIN) & (pys<=TOP_Y_MAX))[0]
    filter_z=np.where((pzs>=TOP_Z_MIN) & (pzs<=TOP_Z_MAX))[0]
    filter_xy=np.intersect1d(filter_x,filter_y)
    filter_xyz=np.intersect1d(filter_xy,filter_z)
    pxs=pxs[filter_xyz]
    pys=pys[filter_xyz]
    pzs=pzs[filter_xyz]
    prs=prs[filter_xyz] 
    lidar= lidar[filter_xyz]

    dist= np.sqrt(np.sum(lidar[:,:3]**2,axis=1))
    evalation_angles = np.arcsin(pzs/dist)/np.pi*180
    keep=np.array([])
    for j in range(len(v16_H)):
        keep=np.union1d(np.where((evalation_angles>=v16_L[j]) & (evalation_angles<=v16_H[j]))[0],keep)
    keep= keep.astype(np.int32)
    # pdb.set_trace()
    pxs=pxs[keep]
    pys=pys[keep]
    pzs=pzs[keep]
    prs=prs[keep] 
    v16= lidar[keep]   

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-TOP_Z_MIN)
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            top[-qxs[i], -qys[i], -2]=prs[i]


    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image=top[:,:,-1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top, density_image, np.array(v16)

def lidar64To16(lidar):
    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]    
    dist= np.sqrt(np.sum(lidar[:,:3]**2,axis=1))
    evalation_angles = np.arcsin(pzs/dist)/np.pi*180
    v16=[]
    num=len(lidar)
    for i in range(num):
        for j in range(len(v16_H)):
            if (evalation_angles[i]>v16_L[j]) and  (evalation_angles[i]<v16_H[j]):
                v16.append(lidar[i])
                break 
    return np.array(v16)

v64 = np.arange(-24.9,2,0.43)
v16_L=np.arange(-15,2,2)-0.2
v16_H=np.arange(-15,2,2)+0.2
v64_intersect_v16=[]
for i in range(len(v64)):
    for j in range(len(v16_H)):
        if (v64[i]>v16_L[j]) and  (v64[i]<v16_H[j]):
            v64_intersect_v16.append(v64[i])
v64_intersect_v16=np.array(v64_intersect_v16)

velodyne = os.path.join(kitti_dir, "velodyne/")
files_list=glob.glob(velodyne+'/*.bin')

###Generate top view data for tracklet.
# train_data_root = "/home/hhs/4T/datasets/dummy_datas_064/seg"
# tracklet_dir = '/home/hhs/4T/datasets/raw data/2011_09_26_drive_0064_sync/2011_09_26/2011_09_26_drive_0064_sync/velodyne_points/data'
# files_list=glob.glob(tracklet_dir+'/*.bin')

lidar_dir = train_data_root+'/lidar16'
top_dir = train_data_root+'/top_70_16'
density_image_dir = train_data_root+'/density_image_70_16'

empty(lidar_dir)
empty(top_dir)
empty(density_image_dir)
makedirs(lidar_dir)
makedirs(top_dir)
makedirs(density_image_dir)

pdb.set_trace()

file=[i.strip().split('/')[-1] for i in files_list]
ind=[int(i.strip().split('.')[0]) for i in file]
num=len(file)

print(num)
L=np.zeros((num))
H=np.zeros((num))
for i in range(num):
    # i=i+7253
    start_time=time()
    filename = velodyne + '/'+file[i]
    print("Processing: ", filename)
    lidar = np.fromfile(filename, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    
    # v16 = lidar64To16(lidar)

    top_new, density_image,v16=lidar_to_top(lidar)

    # fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 1000))
    # draw_lidar(v16, fig=fig)
    # mlab.show(1)   
    # pdb.set_trace() 

    speed=time()-start_time
    print('speed: %0.4fs'%speed)

    np.save(lidar_dir+'/lidar_%05d.npy'%i,v16)
    np.save(top_dir+'/top_70%05d.npy'%ind[i],top_new)
    cv2.imwrite(density_image_dir+'/density_image_70%05d.png'%ind[i],density_image)
   
   
    
    
    
#
# # test
# test = np.load(bird + "000008.npy")

# print(test.shape)
# plt.imshow(test[:,:,8])
# plt.show()



