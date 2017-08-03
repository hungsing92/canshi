from net.common import *
import cv2
from net.utility.draw import *
from net.processing.boxes3d import *

def change_scale(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, scale=0):

	scales=[  1, 1, 0.5,0.7, 0.8, 1.2, 1.6]
	if scales[scale]==1:
		return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d
	print('Scale ratio : %f'%scales[scale])
	width = rgb.shape[1]
	height = rgb.shape[0]
	rgb = cv2.resize(rgb,(int(scales[scale]*width), int(scales[scale]*height)))
	rgbs_norm0 = cv2.resize(rgbs_norm0,(int(scales[scale]*width), int(scales[scale]*height)))
	gt_3dTo2D = gt_3dTo2D*scales[scale]
	gt_box2d = gt_box2d*scales[scale]
	return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d

def flipper(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d):
	print('Flipper')
	rgb = cv2.flip(rgb,1)
	rgbs_norm0 = cv2.flip(rgbs_norm0,1)
	width = rgb.shape[1]
	height = rgb.shape[0]
	gt_box2d[:,[0,2]] = width-gt_box2d[:,[2,0]]-1
	gt_3dTo2D[:,:,0] =width -gt_3dTo2D[:,:,0]-1

	return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d

def regular_2d_box(gt_box2d, width, height):
	gt_box2d_= gt_box2d.copy()
	gt_box2d_[:,0] = np.maximum(np.minimum(gt_box2d_[:,0], width - 1), 0)
	gt_box2d_[:,2] = np.maximum(np.minimum(gt_box2d_[:,2], width - 1), 0)
	gt_box2d_[:,1] = np.maximum(np.minimum(gt_box2d_[:,1], height - 1), 0)
	gt_box2d_[:,3] = np.maximum(np.minimum(gt_box2d_[:,3], height - 1), 0)
	return gt_box2d_

def crop_up(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label, scale = 0):

	scales=[0.1, 0.15, 0.2, 0.25]
	crop_scale = scales[scale]
	print('Crop_scale: %f'%crop_scale)
	width = rgb.shape[1]
	height = rgb.shape[0]
	min_x = min(gt_box2d[:,0])	
	max_x = max(gt_box2d[:,2])

	if 1:
		crop_width = width*(1-2*crop_scale) 
		crop_height = height*(1-2*crop_scale)
		rgb = rgb[int(height*crop_scale):int(height*(1-crop_scale)), int(width*crop_scale):int(width*(1-crop_scale))]
		rgbs_norm0 = rgbs_norm0[int(height*crop_scale):int(height*(1-crop_scale)), int(width*crop_scale):int(width*(1-crop_scale))]

		# pdb.set_trace()
		area = (gt_box2d[:,2]- gt_box2d[:,0])*(gt_box2d[:,3]-gt_box2d[:,1])
		gt_box2d[:,0] = gt_box2d[:,0] - width*crop_scale
		gt_box2d[:,2] = gt_box2d[:,2] - width*crop_scale
		gt_box2d[:,1] = gt_box2d[:,1] - height*crop_scale
		gt_box2d[:,3] = gt_box2d[:,3] - height*crop_scale

		gt_box2d_=regular_2d_box(gt_box2d, crop_width, crop_height)
		area2 = (gt_box2d_[:,2]- gt_box2d_[:,0])*(gt_box2d_[:,3]-gt_box2d_[:,1])
		keep = np.where(area2>area*0.15)[0]
		gt_3dTo2D = gt_3dTo2D[keep]		
		gt_box2d =  gt_box2d[keep]
		gt_label =  gt_label[keep]

		out_of_range = []
		for i in range(len(gt_box2d)):
			if (gt_box2d[i,2]<=0) or (gt_box2d[i,0]>=crop_width) or (gt_box2d[i,3]<=0) or (gt_box2d[i,1]>=crop_height):
				out_of_range.append(i)

		gt_3dTo2D = np.delete(gt_3dTo2D, out_of_range, 0)
		gt_box2d = np.delete(gt_box2d, out_of_range, 0)
		gt_label = np.delete(gt_label, out_of_range, 0)
		gt_3dTo2D[:,:,0] = gt_3dTo2D[:,:,0] - width*crop_scale
		gt_3dTo2D[:,:,1] = gt_3dTo2D[:,:,1] - height*crop_scale

		if np.random.randint(2):
			print('crop and resize')
			rgb = cv2.resize(rgb,(int(width), int(height)))
			rgbs_norm0 = cv2.resize(rgbs_norm0,(int(width), int(height)))
			gt_3dTo2D = (gt_3dTo2D*(1/(1-2*crop_scale))).astype(np.int32)
			gt_box2d = (gt_box2d*(1/(1-2*crop_scale))).astype(np.int32)
		# if len(gt_label)==0:
		# 	pdb.set_trace()
		# img_gt = draw_rgb_projections(rgb, gt_3dTo2D, color=(0,0,255), thickness=1)
		# img_gt = draw_boxes(img_gt,gt_box2d, color=(255,0,255), thickness=1)
		# imshow('crop_img',img_gt)
		# pdb.set_trace()

	return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label

def data_augmentation(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label):
	if len(gt_box2d)==0:
		return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label
	if np.random.randint(5)==0:
		rgb = rgb*0.7
		rgbs_norm0 = rgbs_norm0*0.7
	randInt = np.random.randint(11)
	if randInt<=6:
		rgb, rgbs_norm0, gt_3dTo2D, gt_box2d= change_scale(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d,randInt)
	else:
		randInt = randInt-7
		rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label = crop_up(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label, randInt)
	randInt = np.random.randint(2)
	if randInt==1:
		rgb, rgbs_norm0, gt_3dTo2D, gt_box2d= flipper(rgb, rgbs_norm0, gt_3dTo2D, gt_box2d)

	return rgb, rgbs_norm0, gt_3dTo2D, gt_box2d, gt_label
