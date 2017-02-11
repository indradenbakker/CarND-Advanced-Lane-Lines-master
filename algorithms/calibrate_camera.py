import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob

import pickle


def compute_calibration(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return(undist, mtx, dist)

def undistort_and_warp(directory, dim=(9,6)):

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = [] 
    imgpoints = [] 

    cam_images = glob.glob(directory)

    for fname in cam_images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, dim, None)

        # If not found, return
        if ret == False:
            print("No chessboard corners found")
        else:         
            objpoints.append(objp)
            imgpoints.append(corners)

    shape = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    save_dict = {'mtx': mtx, 'dist': dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)

    return(mtx, dist)