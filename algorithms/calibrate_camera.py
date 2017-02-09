import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compute_calibration(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape,None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return(undist, mtx, dist)

def undistort_and_warp(fname, dim=(9,6)):

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = [] 
    imgpoints = [] 

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])

    ret, corners = cv2.findChessboardCorners(gray, dim, None)

    # If not found, return
    if ret == False:
        print("No chessboard corners found")
        return(img, None, None)

    objpoints.append(objp)
    imgpoints.append(corners)

    img = cv2.drawChessboardCorners(img, dim, corners, ret)
    
    offset = 100

    src = np.float32([corners[0], corners[dim[0]-1], corners[-1], corners[-dim[0]]])
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_undist, mtx, distortion_coef = compute_calibration(img, objpoints, imgpoints)
    warped = cv2.warpPerspective(img_undist, M, img_size)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(img_undist)
    ax2.set_title('Undistorted Image')
    ax3.imshow(warped)
    ax3.set_title('Undistorted & Warped Image')
    return(img_undist, mtx, distortion_coef)