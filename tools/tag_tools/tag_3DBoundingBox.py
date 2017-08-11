import cv2  
import numpy as np   
import time  
import pdb
import glob
import argparse
import h5py
import sys
import os
import shutil

def makedirs(dir):
    if not os.path.isdir(dir): os.makedirs(dir)
def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


def draw_projection(bbs3d, image_):
    thickness=1
    forward_color = (0,255,0)
    bbs3d_ = np.array(bbs3d.copy()).astype(np.int32)
    qs = np.array(bbs3d_)

    for k in range(0,4):
        #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i,j=k,(k+1)%4
        # if length_filter((qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),img):
        #     break
        cv2.line(image_, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        i,j=k+4,(k+1)%4 + 4
        cv2.line(image_, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        i,j=k,k+4
        cv2.line(image_, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    cv2.line(image_, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
    cv2.line(image_, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
    cv2.line(image_, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
    cv2.line(image_, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
    # cv2.line(image_, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
    # cv2.line(image_, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA) 

    # for i in range(8):
    #     # cv2.circle(image_, (qs[i,0],qs[i,1]), 3, forward_color, thickness)
    #     cv2.putText(image_, '%d' % i, (qs[i,0],qs[i,1]), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,255))
    #     pass

    return image_


def click_and_crop_2d(event, x, y, flags, param):
    global bbs

    if event == cv2.EVENT_LBUTTONDOWN:

        bbs.append([x,y,0,0])
         
    elif event == cv2.EVENT_LBUTTONUP:
        bbs[-1][2] = abs(x - bbs[-1][0])            
        bbs[-1][3] = abs(y - bbs[-1][1])
        bbs[-1][0] = min(x, bbs[-1][0])
        bbs[-1][1] = min(y, bbs[-1][1])
        image_ = images[-1][-1].copy()
        cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][0]+bbs[-1][2],bbs[-1][1]+bbs[-1][3]), (255,255,0), 1)
        images[-1].append(image_)
        #cv2.putText(image, 'Upper %d' % id, (bbs[-1][0],bbs[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))
def click_rectangle_backToface_so_far(event, x, y, flags, param):
    global bbs, bbs3d,bbs3ds, temp
    if event == cv2.EVENT_LBUTTONDOWN:   
        if len(bbs3d)==8:
            pass
        else:
            bbs.append([x,y,0,0])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(bbs3d)==8:
            pass
        else:
            width = abs(x - bbs[-1][0])            
            height = abs(y - bbs[-1][1])
            bbs[-1][0] = min(x, bbs[-1][0])
            bbs[-1][1] = min(y, bbs[-1][1])
            bbs[-1][2] = bbs[-1][0]+width
            bbs[-1][3] = bbs[-1][1]+height
    
            image_ = image[-1].copy()
            if len(bbs)%2 == 1:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (0,0,255), 1)
            else:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,0,0), 1)
            print(bbs)
            
            # if len(bbs)%2 == 0:
            # bbs3d = np.zeros([8,2], dtype=np.int32)
            # scale = 2
            bbs3d.append([bbs[0][0],bbs[0][3]])
            bbs3d.append([bbs[0][2],bbs[0][3]])
            bbs3d.append([bbs[0][2]-2,bbs[0][3]-1])
            bbs3d.append([bbs[0][0]+2,bbs[0][3]-1])
            bbs3d.append([bbs[0][0],bbs[0][1]])
            bbs3d.append([bbs[0][2],bbs[0][1]])
            bbs3d.append([bbs[0][2]-2,bbs[0][1]+2])
            bbs3d.append([bbs[0][0]+2,bbs[0][1]+2])
            
            temp = bbs3d.copy()
            image_=draw_projection(temp, image_)
            image.append(image_)            
            # bbs3d.append(bbs3d_)
            # right_up_flag = not right_up_flag
    elif event == cv2.EVENT_RBUTTONUP:
        if len(bbs)>0:
            bbs3d=[]  
            bbs.pop()
            if len(image)>1:
                image.pop()

def click_rectangle_faceToface_so_far(event, x, y, flags, param):
    global bbs, bbs3d,bbs3ds, temp
    if event == cv2.EVENT_LBUTTONDOWN:   
        if len(bbs3d)==8:
            pass
        else:
            bbs.append([x,y,0,0])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(bbs3d)==8:
            pass
        else:
            width = abs(x - bbs[-1][0])            
            height = abs(y - bbs[-1][1])
            bbs[-1][0] = min(x, bbs[-1][0])
            bbs[-1][1] = min(y, bbs[-1][1])
            bbs[-1][2] = bbs[-1][0]+width
            bbs[-1][3] = bbs[-1][1]+height
    
            image_ = image[-1].copy()
            # cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,255,0), 1)
            if len(bbs)%2 == 1:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (0,0,255), 1)
            else:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,0,0), 1)            
            print(bbs)
            
            # if len(bbs)%2 == 0:
            # bbs3d = np.zeros([8,2], dtype=np.int32)
            scale = 2
            bbs3d.append([bbs[0][2]-scale,bbs[0][3]-scale])
            bbs3d.append([bbs[0][0]-scale,bbs[0][3]-scale])            
            bbs3d.append([bbs[0][0],bbs[0][3]])
            bbs3d.append([bbs[0][2],bbs[0][3]])

            bbs3d.append([bbs[0][2]-scale,bbs[0][1]])
            bbs3d.append([bbs[0][0]-scale,bbs[0][1]])
            bbs3d.append([bbs[0][0],bbs[0][1]+scale])
            bbs3d.append([bbs[0][2],bbs[0][1]+scale])           
            temp = bbs3d.copy()
            image_=draw_projection(temp, image_)  
            image.append(image_)          
            # bbs3d.append(bbs3d_)
            # right_up_flag = not right_up_flag
    elif event == cv2.EVENT_RBUTTONUP:
        if len(bbs)>0:
            bbs3d=[]  
            bbs.pop()
            if len(image)>1:
                image.pop()


def click_rectangle_backToface(event, x, y, flags, param):
    global bbs, bbs3d,bbs3ds, temp

    if event == cv2.EVENT_LBUTTONDOWN:

        
        if len(bbs3d)==8:
            # bbs3ds.append(bbs3d)
            # bbs3d = []
            # bbs = []
            pass
        else:
            bbs.append([x,y,0,0])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(bbs3d)==8:
            pass
        else:
            width = abs(x - bbs[-1][0])            
            height = abs(y - bbs[-1][1])
            bbs[-1][0] = min(x, bbs[-1][0])
            bbs[-1][1] = min(y, bbs[-1][1])
            bbs[-1][2] = bbs[-1][0]+width
            bbs[-1][3] = bbs[-1][1]+height
    
            image_ = image[-1].copy()
            # cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,255,0), 1)
            if len(bbs)%2 == 1:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (0,0,255), 1)
            else:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,0,0), 1)            
            print(bbs)
            # image.append(image_)  
            if len(bbs)%2 == 0:
                # bbs3d = np.zeros([8,2], dtype=np.int32)
                bbs3d.append([bbs[0][0],bbs[0][3]])
                bbs3d.append([bbs[0][2],bbs[0][3]])
                bbs3d.append([bbs[1][2],bbs[1][3]])
                bbs3d.append([bbs[1][0],bbs[1][3]])
                bbs3d.append([bbs[0][0],bbs[0][1]])
                bbs3d.append([bbs[0][2],bbs[0][1]])
                bbs3d.append([bbs[1][2],bbs[1][1]])
                bbs3d.append([bbs[1][0],bbs[1][1]])
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]
                bbs3d[7][0] = bbs3d[3][0]
                
                temp = bbs3d.copy()

                image_=draw_projection(temp, image_)   
                for i in range(8):
                    cv2.putText(image_, '%d' % i, (bbs3d[i][0],bbs3d[i][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,255))
            image.append(image_)         
                # bbs3d.append(bbs3d_)
            # right_up_flag = not right_up_flag
    elif event == cv2.EVENT_RBUTTONUP:
        if len(bbs)>0:

            bbs3d=[]  
            bbs.pop()

            if len(image)>1:
                image.pop()

def click_rectangle_faceToface(event, x, y, flags, param):
    global bbs, bbs3d,bbs3ds, temp

    if event == cv2.EVENT_LBUTTONDOWN:       
        if len(bbs3d)==8:

            pass
        else:
            bbs.append([x,y,0,0])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(bbs3d)==8:
            pass
        else:
            width = abs(x - bbs[-1][0])            
            height = abs(y - bbs[-1][1])
            bbs[-1][0] = min(x, bbs[-1][0])
            bbs[-1][1] = min(y, bbs[-1][1])
            bbs[-1][2] = bbs[-1][0]+width
            bbs[-1][3] = bbs[-1][1]+height
    
            image_ = image[-1].copy()
            # cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,255,0), 1)
            if len(bbs)%2 == 1:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (0,0,255), 1)
            else:
                cv2.rectangle(image_, (bbs[-1][0],bbs[-1][1]), (bbs[-1][2],bbs[-1][3]), (255,0,0), 1)            
            print(bbs)
            # image.append(image_)  
            if len(bbs)%2 == 0:
                # bbs3d = np.zeros([8,2], dtype=np.int32)
                bbs3d.append([bbs[0][0],bbs[0][3]])
                bbs3d.append([bbs[0][2],bbs[0][3]])
                bbs3d.append([bbs[1][2],bbs[1][3]])
                bbs3d.append([bbs[1][0],bbs[1][3]])
                bbs3d.append([bbs[0][0],bbs[0][1]])
                bbs3d.append([bbs[0][2],bbs[0][1]])
                bbs3d.append([bbs[1][2],bbs[1][1]])
                bbs3d.append([bbs[1][0],bbs[1][1]])
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]
                bbs3d[7][0] = bbs3d[3][0]
  
                temp = bbs3d.copy()
                for j in range(4):
                    temp[j*2] = bbs3d[j*2+1]
                    temp[j*2+1] = bbs3d[j*2]                
                # right_up_flag = False
                image_=draw_projection(temp, image_) 
                for i in range(8):
                    cv2.putText(image_, '%d' % i, (bbs3d[i][0],bbs3d[i][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,255))                
            image.append(image_)           
                # bbs3d.append(bbs3d_)
            # right_up_flag = not right_up_flag
    elif event == cv2.EVENT_RBUTTONUP:
        if len(bbs)>0:
            bbs3d=[]  
            bbs.pop()

            if len(image)>1:
                image.pop()

def click_point_side_backToface(event, x, y, flags, param):
    global bbs3d, temp
    if event == cv2.EVENT_LBUTTONDOWN :
        if len(bbs3d)==8:
            pass
        else: 

            bbs3d.append(np.array([int(x),int(y)]))
            image_ = image[-1].copy()
            if len(bbs3d)==1:
                cv2.circle(image_, (bbs3d[0][0], bbs3d[0][1]),1, (00,00,255), 1, cv2.LINE_AA)
            # images[-1].append(image_)
            if len(bbs3d)<=3 and len(bbs3d)>0:
                for k in range(0,len(bbs3d)-1):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                if len(bbs3d)<3:
                    image.append(image_)
                # pdb.set_trace()
            if len(bbs3d)==3:
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[0][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[0][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                image.append(image_)
            if len(bbs3d)==4:
                for k in range(0,len(bbs3d)):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                image.append(image_)       
            if len(bbs3d)==5:    
                bbs3d[4][0] = bbs3d[0][0]
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[1][0]+(bbs3d[4][0]-bbs3d[0][0]), bbs3d[1][1]+(bbs3d[4][1]-bbs3d[0][1])]
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==6:       
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]         
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[5][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[5][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[5][0],bbs3d[5][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==7:   
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]             
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[2][0],bbs3d[2][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[3][0]+(bbs3d[6][0]-bbs3d[2][0]), bbs3d[3][1]+(bbs3d[6][1]-bbs3d[2][1])]
                cv2.line(image_, (bbs3d[3][0],bbs3d[3][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
    
            if len(bbs3d)==8:
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]
                bbs3d[7][0] = bbs3d[3][0]
    
                if bbs3d[0][1]> bbs3d[3][1]:
                    if bbs3d[0][0]>bbs3d[1][0]:
                        temp = bbs3d.copy()
                        for j in range(4):
                            temp[j*2] = bbs3d[j*2+1]
                            temp[j*2+1] = bbs3d[j*2]
                    else:
                        temp = bbs3d.copy()

                image_=draw_projection(temp, image_)
                for i in range(8):
                    cv2.putText(image_, '%d' % i, (bbs3d[i][0],bbs3d[i][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,255))
                image.append(image_)

    if event == cv2.EVENT_RBUTTONDOWN :
        if len(bbs3d)>0:
            bbs3d.pop()
            if len(image)>1:
                image.pop()

def click_point_side_faceToface(event, x, y, flags, param):
    global bbs3d, temp
    if event == cv2.EVENT_LBUTTONDOWN :
        if len(bbs3d)==8:
            # bbs3ds.append(temp)
            # bbs3d = []
            # bbs = []
            # temp = []
            # continue
            pass
        else:
            bbs3d.append(np.array([int(x),int(y)]))
            image_ = image[-1].copy()
            if len(bbs3d)==1:
                cv2.circle(image_, (bbs3d[0][0], bbs3d[0][1]),1, (00,00,255), 1, cv2.LINE_AA)
            # images[-1].append(image_)
            if len(bbs3d)<=3 and len(bbs3d)>0:
                for k in range(0,len(bbs3d)-1):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                if len(bbs3d)<3:
                    image.append(image_)
                # pdb.set_trace()
            if len(bbs3d)==3:
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[0][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[0][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                image.append(image_)
            if len(bbs3d)==4:
                for k in range(0,len(bbs3d)):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                image.append(image_)       
            if len(bbs3d)==5:    
                bbs3d[4][0] = bbs3d[0][0]
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[1][0]+(bbs3d[4][0]-bbs3d[0][0]), bbs3d[1][1]+(bbs3d[4][1]-bbs3d[0][1])]
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==6:       
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]         
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[5][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[5][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[5][0],bbs3d[5][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==7:   
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]             
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[2][0],bbs3d[2][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[3][0]+(bbs3d[6][0]-bbs3d[2][0]), bbs3d[3][1]+(bbs3d[6][1]-bbs3d[2][1])]
                cv2.line(image_, (bbs3d[3][0],bbs3d[3][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
    
            if len(bbs3d)==8:
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]
                bbs3d[7][0] = bbs3d[3][0]
                #face to face leftside
                if bbs3d[0][0]<bbs3d[1][0]:
                    temp = bbs3d.copy()
                    for j in range(2):
                        temp[j*4] = bbs3d[j*4+2]
                        temp[j*4+1] = bbs3d[j*4+3]
                        temp[j*4+2] = bbs3d[j*4]
                        temp[j*4+3] = bbs3d[j*4+1]
                else:
                    temp = bbs3d.copy()
                    for j in range(2):
                        temp[j*4] = bbs3d[j*4+3]
                        temp[j*4+1] = bbs3d[j*4+2]
                        temp[j*4+3] = bbs3d[j*4]
                        temp[j*4+2] = bbs3d[j*4+1]                    
    
                image_=draw_projection(temp, image_)
                for i in range(8):
                    cv2.putText(image_, '%d' % i, (bbs3d[i][0],bbs3d[i][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,255))
                image.append(image_)

    if event == cv2.EVENT_RBUTTONDOWN :
        if len(bbs3d)>0:
            bbs3d.pop()
            if len(image)>1:
                image.pop()

def click_point_side(event, x, y, flags, param):
    global bbs3d, temp
    if event == cv2.EVENT_LBUTTONDOWN :
        if len(bbs3d)==8:
            pass
        else: 

            bbs3d.append(np.array([int(x),int(y)]))
            image_ = image[-1].copy()
            if len(bbs3d)==1:
                cv2.circle(image_, (bbs3d[0][0], bbs3d[0][1]),1, (00,00,255), 1, cv2.LINE_AA)
            # images[-1].append(image_)
            if len(bbs3d)<=3 and len(bbs3d)>0:
                for k in range(0,len(bbs3d)-1):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                if len(bbs3d)<3:
                    image.append(image_)
                # pdb.set_trace()
            if len(bbs3d)==3:
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[0][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[0][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                image.append(image_)
            if len(bbs3d)==4:
                for k in range(0,len(bbs3d)):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                image.append(image_)       
            if len(bbs3d)==5:    
                bbs3d[4][0] = bbs3d[0][0]
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[1][0]+(bbs3d[4][0]-bbs3d[0][0]), bbs3d[1][1]+(bbs3d[4][1]-bbs3d[0][1])]
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==6:       
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]         
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[2][0]+(bbs3d[5][0]-bbs3d[1][0]), bbs3d[2][1]+(bbs3d[5][1]-bbs3d[1][1])]
                cv2.line(image_, (bbs3d[2][0],bbs3d[2][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[5][0],bbs3d[5][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
            if len(bbs3d)==7:   
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]             
                for k in range(0,4):
                    i,j=k,(k+1)%4
                    cv2.line(image_, (bbs3d[i][0],bbs3d[i][1]), (bbs3d[j][0],bbs3d[j][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[0][0],bbs3d[0][1]), (bbs3d[4][0],bbs3d[4][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[1][0],bbs3d[1][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[5][0],bbs3d[5][1]), color, 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (bbs3d[2][0],bbs3d[2][1]), color, 1, cv2.LINE_AA)
                auxiliary_point3 = [bbs3d[3][0]+(bbs3d[6][0]-bbs3d[2][0]), bbs3d[3][1]+(bbs3d[6][1]-bbs3d[2][1])]
                cv2.line(image_, (bbs3d[3][0],bbs3d[3][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[6][0],bbs3d[6][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
                cv2.line(image_, (bbs3d[4][0],bbs3d[4][1]), (auxiliary_point3[0] ,auxiliary_point3[1]), (255,100,0), 1, cv2.LINE_AA)
    
                image.append(image_)
    
            if len(bbs3d)==8:
                bbs3d[4][0] = bbs3d[0][0]
                bbs3d[5][0] = bbs3d[1][0]
                bbs3d[6][0] = bbs3d[2][0]
                bbs3d[7][0] = bbs3d[3][0]

                temp = bbs3d.copy()

                image_=draw_projection(temp, image_)
                for i in range(8):
                    cv2.putText(image_, '%d' % i, (bbs3d[i][0],bbs3d[i][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,255))
                image.append(image_)

    if event == cv2.EVENT_RBUTTONDOWN :
        if len(bbs3d)>0:
            bbs3d.pop()
            if len(image)>1:
                image.pop()



def display(img, bbs3ds):
    image_ = img.copy()
    if len(bbs3ds) ==0:
        return image_
    bbs3ds_ = np.array(bbs3ds.copy())
    # bbs3ds_[:,:,0] = bbs3ds_[:,:,0] - int(width*up_scale)
    # bbs3ds_[:,:,1] = bbs3ds_[:,:,1] - int(height*up_scale)
    for i in range(len(bbs3ds_)):
        bbs3d_ = bbs3ds_[i]
        # pdb.set_trace()
        if len(bbs3d_)>0:
            image_=draw_projection(bbs3d_, image_) 
            cv2.imshow("image", image_)
    return image_


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-s", "--save_path", required=True, help="Path to save tag")
    ap.add_argument("-b", "--begin", default= None ,help="image name to begin tag")
    args = vars(ap.parse_args())
    img_fn = args["image"]
    image_begin = args["begin"]
    print('bdgin from : %s.png'%image_begin)

    save_path = args["save_path"]
    save_image = save_path+"/tag_image"
    save_label = save_path+"/tag_label"
    # empty(save_image)
    # empty(save_label)

    makedirs(save_image)
    makedirs(save_label)    

    image_list = np.sort(glob.glob(img_fn+'/*.png'))
    file_name = [i.strip().split('/')[-1].split('.')[0] for i in image_list]
    if image_begin in file_name:
        begin = file_name.index(image_begin)
    else:
        begin = 0
    cv2.namedWindow('image') 
    for n in range(len(image_list)):
        n = n+ begin
        img=cv2.imread(image_list[n]) 
        clone = img.copy()

        width = img.shape[1]
        height = img.shape[0]

        up_scale = 0.5
        up_scale_width = int(width*(1+2*up_scale))
        up_scale_height = int(height*(1+2*up_scale))

        # emptyImage = np.zeros(img.shape, np.uint8) 
        up_image= np.zeros([int(height*(1+2*up_scale)), int(width*(1+2*up_scale)), 3],dtype=np.uint8)
        up_image[int(height*(up_scale)):int(height*(1+1*up_scale)), int(width*(up_scale)):int(width*(1+1*up_scale)),:] = clone
        clone = up_image.copy()
        color=(0,0,255)

        for i in np.arange(0,up_scale_width,30):
            i = int(i)
            cv2.line(clone, (i,0), (i, up_scale_height-1), (100,100,100), 1, cv2.LINE_AA)
        for j in np.arange(0, up_scale_height, 30):
            j= int(j)
            cv2.line(clone, (0,j), (up_scale_width-1,j), (100,100,100), 1, cv2.LINE_AA)
        image = []
        image.append(clone)
        
        bbs2d = []
        bbs3d = []
        bbs3ds = []
        temp=[]
        backTofaceflag=False
        bbs =[]
        
        # cv2.setMouseCallback("image", click_point_side)
        while True:

            cv2.imshow("image", image[-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                image = []
                bbs = []
                bbs3d = []
                bbs3ds = []
                temp = []
                image.append(display(clone, bbs3ds))
            elif key == ord("n"):
                bbs3ds.append(temp)
                bbs3d = []
                bbs =[]
                image = []
                image.append(display(clone, bbs3ds))
            elif key == ord("d"):
                # bbs3ds.pop()
                bbs3d = []
                bbs =[]
                image = []
                image.append(display(clone, bbs3ds))  
            elif key == ord("D"):
                bbs3ds.pop()
                bbs3d = []
                bbs =[]
                image = []
                image.append(display(clone, bbs3ds))                               
            elif key == ord("s"):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                    temp=[]
                image = []
                image.append(display(clone, bbs3ds))    
                bbs3d = []
                break             
            elif key == ord('q'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs3d = []
                bbs =[]
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_point_side_backToface) 
            elif key == ord('w'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_point_side_faceToface)
            elif key == ord('e'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_point_side)                
            elif key == ord('1'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_rectangle_backToface)
            elif key == ord('2'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_rectangle_faceToface)   
            elif key == ord('3'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_rectangle_backToface_so_far)                   
            elif key == ord('4'):
                if len(bbs3d)==8:
                    bbs3ds.append(temp)
                bbs =[]
                bbs3d = []
                image = []
                image.append(display(clone, bbs3ds))
                cv2.setMouseCallback('image',click_rectangle_faceToface_so_far)         

        image_ = img.copy()
        if len(bbs3ds)>0:
            bbs3ds_ = np.array(bbs3ds.copy())
            bbs3ds_[:,:,0] = bbs3ds_[:,:,0] - int(width*up_scale)
            bbs3ds_[:,:,1] = bbs3ds_[:,:,1] - int(height*up_scale)
            for i in range(len(bbs3ds_)):
                bbs3d_ = bbs3ds_[i]
                # pdb.set_trace()
                if len(bbs3d_)>0:
                    image_=draw_projection(bbs3d_, image_) 
                    cv2.imshow("image", image_)
                    # pdb.set_trace()
            cv2.imwrite(save_image+'/'+file_name[n]+'_tag.png',image_)
            h5_fn = save_label +'/'+file_name[n]+'.h5'
            with h5py.File(h5_fn,'w') as h5f:
                h5f.create_dataset('label', data=bbs3ds)       
        cv2.destroyAllWindows()       

