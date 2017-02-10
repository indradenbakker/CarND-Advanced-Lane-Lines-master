## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Below I will describe point by point the steps taken in this algorithm to detect the lane lines. 

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibaration
Camera calibration is done by the script `calibrate_camera.py` located in the algorithms folder. 

### Discussion
This alogrithm has problems identifying the lane lines in the challenge videos, because of the contrast and the fact that the lane lines are not as clearly visible as in the project video. Possible improvements:

* Remove outliers before fitting the polynomials.
* Implement region of interest (ROI)
* Tweak thresholds
* Compare frame with previous frame(s) to reduce sudden changes
