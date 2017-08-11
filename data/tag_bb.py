# import the necessary packages
import argparse
import cv2
import h5py
import numpy as np


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
bbs = []
x_upper = True
id = 1

def click_and_crop(event, x, y, flags, param):
    global bbs, x_upper, id

    if event == cv2.EVENT_LBUTTONDOWN:
        if x_upper:
            bbs.append([x,y,0,0, 0,0,0,0])
        else:
            bbs[-1][4] = x
            bbs[-1][5] = y
            
    elif event == cv2.EVENT_LBUTTONUP:
        if x_upper:
            bbs[-1][2] = abs(x - bbs[-1][0])            
            bbs[-1][3] = abs(y - bbs[-1][1])
            bbs[-1][0] = min(x, bbs[-1][0])
            bbs[-1][1] = min(y, bbs[-1][1])
            cv2.rectangle(image, (bbs[-1][0],bbs[-1][1]), (bbs[-1][0]+bbs[-1][2],bbs[-1][1]+bbs[-1][3]), (0,0,255), 2)
            #cv2.putText(image, 'Upper %d' % id, (bbs[-1][0],bbs[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))
        else:
            bbs[-1][6] = abs(x - bbs[-1][4])
            bbs[-1][7] = abs(y - bbs[-1][5])
            bbs[-1][4] = min(x, bbs[-1][4])
            bbs[-1][5] = min(y, bbs[-1][5])
            cv2.rectangle(image, (bbs[-1][4],bbs[-1][5]), (bbs[-1][4]+bbs[-1][6],bbs[-1][5]+bbs[-1][7]), (0,255,0), 2)
            cv2.putText(image, 'Body %d' % id, (bbs[-1][4],bbs[-1][5]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0))
            
            
        cv2.imshow("image", image)        
        x_upper = not x_upper
        


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-s", "--save_path", required=True, help="Path to save tag")
    args = vars(ap.parse_args())
    
    img_fn = args["image"]
    print(args["save_path"])
    image = cv2.imread(img_fn)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            image = clone.copy()
            bbs = []
            x_upper = True 
            id = 1
        elif key == ord("n"):
            id = id+1
        elif key == ord("s"):
            break
    
    cv2.destroyAllWindows()
    
    print( np.array(bbs))
     
    h5_fn = img_fn[:-4].replace('img','bb') + '.h5'
 
    with h5py.File(h5_fn,'w') as h5f:
        h5f.create_dataset('label', data=bbs)
        
    img_bb_fn = img_fn.replace('img','imgbb')
    cv2.imwrite(img_bb_fn,image)