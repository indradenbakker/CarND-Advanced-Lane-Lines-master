import numpy as np
import cv2

def warp_image(img, src_points, dst_points):
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, M_inv