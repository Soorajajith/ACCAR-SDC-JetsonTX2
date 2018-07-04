import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')


#fig, axs = plt.subplots(5,4, figsize=(16,11))
#Figure(1152x792)
#array([5][4] for axs)
#fig.subplots_adjust(hspace = 0.2, wspace =0.001)
#axs = axs.ravel()
#array(5*4=20)

# Step through the list of images and search for chessboard corners in each one
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
    #ret bool, cornoers corner 좌표.

    # If found, add object points, image points to the lists
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#ret = 0.9231135540074387
#mtx = array(3,3)
#dist = [[-0.23157149 -0.1200054  -0.00118338  0.00023305  0.15641575]]
#rvecs = array(17,3,1)
#tvecs = array((17, 3, 1))

#Finds the camera intrinsic and extrinsic parameters
#from several views of a calibration pattern.



def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
