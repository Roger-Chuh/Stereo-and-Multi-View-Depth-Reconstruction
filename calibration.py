# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 07:13:21 2016

@author: xumr
"""

import numpy as np
import cv2

class calibration:
    
    def __init__(self):
        self.criteria_CameraCalibration = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #迭代停止的模式选择，格式为(type,max_iter,epsilon)
        #EPS:精确度（误差）满足epsilon停止
        #Max iter：迭代次数超过max_iter停止
        self.criteria_StereoCalibration = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, 1e-5)
        self.flag_CameraCalibration = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4
        self.flags_StereoCalibration = cv2.CALIB_FIX_INTRINSIC
        self.image_size = (720L, 480L)
        self.square_size = 37
        self.objp = np.zeros((6*9,3), np.float32)
        self.objp[:,:2] = self.square_size*np.mgrid[0:9,0:6].T.reshape(-1,2)
        


    def CameraCalibration(self,side):
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        for number in range(40,78): #只给一个相机做calibration
            fname = side+str(number)+'.jpg'
            print "Finding chessboard corners in image %s" %fname
            img = cv2.imread(fname)
        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            
            
            if ret == True:

                objpoints.append(self.objp)
        
                #cornerSubPix: increase the accuracy of the corners we found
                #Refines the corner locations
                cv2.cornerSubPix (gray, corners,(11,11),(-1,-1), self.criteria_CameraCalibration)
                imgpoints.append(corners.reshape(-1,2))
        
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, True)
                #cv2.imshow(fname,img)
                #cv2.waitKey(0)
        
            else:
                print('Cannot find chessboard in %s') %fname
        
        
        print "Doing Camera Calibration for the %s camera" %side

        (ret,cam_mats,dist_coefs, rvecs,tvecs) = \
            cv2.calibrateCamera(objpoints,imgpoints,self.image_size,flags=self.flag_CameraCalibration)
            
        return objpoints, imgpoints, cam_mats,dist_coefs, rvecs,tvecs
        
        
    def StereoCalibration(self,objpoints,imgpoints1,imgpoints2,cam_mats1,cam_mats2,dist_coefs1,dist_coefs2):
        
        (retval, cam_mats1, dist_coefs1,cam_mats2, dist_coefs2, R, T, E, F) = \
            cv2.stereoCalibrate(objpoints,imgpoints1,imgpoints2,imageSize=self.image_size,
                                cameraMatrix1=cam_mats1,distCoeffs1=dist_coefs1,
                                cameraMatrix2=cam_mats2,distCoeffs2=dist_coefs2,
                                criteria=self.criteria_StereoCalibration,flags=self.flags_StereoCalibration)#[1:]
                                      
        return R,T