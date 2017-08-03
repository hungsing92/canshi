import _init_paths
import matplotlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import os
import pdb
from net.utility.file import *

data_root='/home/hhs/4T/hongsheng/2dTo3D/faster_rcnn/examples/'
save_root = data_root+'Demo_Vedio/'
makedirs(save_root)
folder_name = 'result_crop2016_0306_110310_227'
files_list=glob.glob(data_root+folder_name+'/*.png')
index=np.array([file_index.strip().split('/')[-1].split('.')[0] for file_index in files_list ])
num_frames=len(files_list)

video_path=save_root+folder_name+'.avi'
fps = 6
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
img=cv2.imread(data_root+folder_name+'/%05d.png'%2)
videoWriter = cv2.VideoWriter(video_path, fourcc,fps,(img.shape[1], img.shape[0]))#最后一个是保存图片的尺寸
# fig, axes = plt.subplots(2, 1, figsize=(16, 12))
# ax0, ax1 = axes.ravel()

for i in range(num_frames):

	img=cv2.imread(data_root+folder_name+'/%05d.png'%(i+1))
	videoWriter.write(img)
	# cv2.waitKey(1)	
videoWriter.release()