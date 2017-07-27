import _init_paths
import matplotlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import os
import pdb

data_root='/home/hhs/4T/datasets/dummy_datas_064/seg'
files_list=glob.glob(data_root+'/mayavi_fig/*.png')
index=np.array([file_index.strip().split('/')[-1][10:10+5] for file_index in files_list ])
num_frames=len(files_list)

video_path=data_root+'/result_video.avi'
fps = 8
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(video_path,fourcc,fps,(1600,1200))#最后一个是保存图片的尺寸
# fig, axes = plt.subplots(2, 1, figsize=(16, 12))
# ax0, ax1 = axes.ravel()

for i in range(num_frames):
	fig, axes = plt.subplots(2, 1, figsize=(16, 12))
	ax0, ax1 = axes.ravel()
	mFig = mpimg.imread(data_root+'/mayavi_fig/mayavi_%05d.png'%i)
	rgbFig=mpimg.imread(data_root+'/result_rgb/rgb_%05d.png'%i)
	ax0.imshow(mFig)
	ax0.axis('off')
	ax1.imshow(rgbFig)
	ax1.axis('off')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig(data_root+'/result_video_img/result_video_img_%05d.png'%i)
	plt.close()
	img=cv2.imread(data_root+'/result_video_img/result_video_img_%05d.png'%i)
	videoWriter.write(img)
	# cv2.waitKey(1)	
videoWriter.release()