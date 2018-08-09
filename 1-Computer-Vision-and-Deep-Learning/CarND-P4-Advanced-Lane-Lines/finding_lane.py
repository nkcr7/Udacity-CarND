import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
from pathlib import Path

########## camera calibration ###############
if Path('./calibration.p').exists():
    with open('./calibration.p', 'rb') as handle:
        calibration = pickle.load(handle)
        ret = calibration['ret']
        mtx = calibration['mtx']
        dist = calibration['dist']
        rvecs = calibration['rvecs']
        tvecs = calibration['tvecs']
else:
    cal_images = glob.glob('./camera_cal/*.jpg')

    nx = 9
    ny = 6

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    obj_points = []
    image_points = []

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in cal_images:
        img = cv2.imread(image_path) # read in BGR mode
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray,(nx,ny))

        if ret is True:
            obj_points.append(objp)
            corner2 = cv2.cornerSubPix(gray, corner, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img, (nx,ny),corner2,ret)
            image_points.append(corner2)
            cv2.imshow('img', img)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, gray.shape[::-1],None,None)

    calibration={}
    calibration['ret'] = ret
    calibration['mtx'] = mtx
    calibration['dist'] = dist
    calibration['rvecs'] = rvecs
    calibration['tvecs'] = tvecs
    with open('./calibration.p', 'wb') as handle:
        pickle.dump(calibration, handle)



############### perspective transform ##############
perspective_images = cv2.imread('./test_images/straight_lines1.jpg')
perspective_src = cv2.undistort(perspective_images, mtx, dist, None, mtx)

src_region = np.float32([[441,560], [849,560], [215,719], [1090,719]])
dst_region = np.float32([[250,630], [900,630], [250,719], [900,719]])
dsize = perspective_src.shape[0:2][::-1]
print(dsize)

M = cv2.getPerspectiveTransform(src_region, dst_region)
inv_M = cv2.getPerspectiveTransform(dst_region, src_region)
after_perspective = cv2.warpPerspective(perspective_src, M, dsize, flags=cv2.INTER_LINEAR)
inv_perspective = cv2.warpPerspective(after_perspective, inv_M, dsize, flags=cv2.INTER_LINEAR)


cv2.imshow('after_perspective', after_perspective)
cv2.waitKey(1000000)


########## finding lanes ###############
# test_images = glob.glob('./test_images/*.jpg')
#
# for image_path in test_images:
#     img = cv2.imread(image_path)
#     dst = cv2.undistort(img, mtx, dist, None, mtx)
#     cv2.imshow('dst', dst)
#     cv2.waitKey(1000)

    