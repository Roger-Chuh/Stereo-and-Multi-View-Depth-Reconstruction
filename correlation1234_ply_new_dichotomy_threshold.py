# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 07:13:21 2016

@author: xumr
"""

import numpy as np
import cv2
from calibration import calibration
from correlation import correlation
import time

start =time.clock()



'''Initialization'''
sample = 'Sample3'

calibration = calibration()
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = {"t_left": [], "t_right": [], "b_left": [], "b_right": []} # 2d points in image plane.
image_size = (720L, 480L)
#: Camera matrices (M)
cam_mats = {"t_left": None, "t_right": None, "b_left": None, "b_right": None}
#: Distortion coefficients (D)
dist_coefs = {"t_left": None, "t_right": None, "b_left": None, "b_right": None}
rvecs = {"t_left": None, "t_right": None, "b_left": None, "b_right": None} #from camer to image?
tvecs = {"t_left": None, "t_right": None, "b_left": None, "b_right": None}
sides = ['t_left','t_right','b_left','b_right']
window_size = 20
Z = range(20, 3001, 10)
pixel_step = 5


'''Camera Calibration'''

for side in sides:
    (objpoints, imgpoints[side], cam_mats[side],dist_coefs[side], rvecs[side],tvecs[side]) = \
        calibration.CameraCalibration(side)



'''Stereo Calibration'''

print "Doing Stereo Calibration for camera 12"
(R12,T12) = calibration.StereoCalibration(objpoints,imgpoints['t_left'],imgpoints['t_right'],
                                            cam_mats['t_left'],cam_mats['t_right'],dist_coefs['t_left'],dist_coefs['t_right'])
print "Doing Stereo Calibration for camera 13"
(R13,T13) = calibration.StereoCalibration(objpoints,imgpoints['t_left'],imgpoints['b_left'],
                                            cam_mats['t_left'],cam_mats['b_left'],dist_coefs['t_left'],dist_coefs['b_left'])    
print "Doing Stereo Calibration for camera 14"
(R14,T14) = calibration.StereoCalibration(objpoints,imgpoints['t_left'],imgpoints['b_right'],
                                            cam_mats['t_left'],cam_mats['b_right'],dist_coefs['t_left'],dist_coefs['b_right'])
                              


'''Read Images'''

t_left = cv2.imread(sample+'_t_left.jpeg')
t_right = cv2.imread(sample+'_t_right.jpeg')
b_left = cv2.imread(sample+'_b_left.jpeg')
b_right = cv2.imread(sample+'_b_right.jpeg')

    
    
'''Zmap'''

print "computing Z values"
Zmap = []
correlation = correlation()
for v1 in range(window_size,480-window_size,pixel_step):
    for u1 in range(window_size,720-window_size,pixel_step):
        '''80mm'''
        Z = range(80, 3081, 80)
        (correlation12,u2max_right,v2max_right) = correlation.correlation(R12,T12,t_left,t_right,cam_mats['t_left'],cam_mats['t_right'],u1,v1,window_size,Z)
        (correlation13,u2max_downl,v2max_downl) = correlation.correlation(R13,T13,t_left,b_left,cam_mats['t_left'],cam_mats['b_left'],u1,v1,window_size,Z)
        (correlation14,u2max_downr,v2max_downr) = correlation.correlation(R14,T14,t_left,b_right,cam_mats['t_left'],cam_mats['b_right'],u1,v1,window_size,Z)
        
        correlationAll = np.array([correlation12,correlation13,correlation14])
        correlationMean = np.mean(correlationAll,axis=0)
        correlationMean = correlationMean.tolist()
        Z_final = Z[correlationMean.index(max(correlationMean))]
        
        
        '''40mm'''
        Z = range(Z_final-40, Z_final+41, 40)
        (correlation12,u2max_right,v2max_right) = correlation.correlation(R12,T12,t_left,t_right,cam_mats['t_left'],cam_mats['t_right'],u1,v1,window_size,Z)
        (correlation13,u2max_downl,v2max_downl) = correlation.correlation(R13,T13,t_left,b_left,cam_mats['t_left'],cam_mats['b_left'],u1,v1,window_size,Z)
        (correlation14,u2max_downr,v2max_downr) = correlation.correlation(R14,T14,t_left,b_right,cam_mats['t_left'],cam_mats['b_right'],u1,v1,window_size,Z)
        
        correlationAll = np.array([correlation12,correlation13,correlation14])
        correlationMean = np.mean(correlationAll,axis=0)
        correlationMean = correlationMean.tolist()
        Z_final = Z[correlationMean.index(max(correlationMean))]
        
        if Z_final<2500:
            '''20mm'''
            Z = range(Z_final-20, Z_final+21, 20)
            (correlation12,u2max_right,v2max_right) = correlation.correlation(R12,T12,t_left,t_right,cam_mats['t_left'],cam_mats['t_right'],u1,v1,window_size,Z)
            (correlation13,u2max_downl,v2max_downl) = correlation.correlation(R13,T13,t_left,b_left,cam_mats['t_left'],cam_mats['b_left'],u1,v1,window_size,Z)
            (correlation14,u2max_downr,v2max_downr) = correlation.correlation(R14,T14,t_left,b_right,cam_mats['t_left'],cam_mats['b_right'],u1,v1,window_size,Z)
            
            correlationAll = np.array([correlation12,correlation13,correlation14])
            correlationMean = np.mean(correlationAll,axis=0)
            correlationMean = correlationMean.tolist()
            Z_final = Z[correlationMean.index(max(correlationMean))]
            
            if Z_final<1500:
                '''10mm'''
                Z = range(Z_final-10, Z_final+11, 10)
                (correlation12,u2max_right,v2max_right) = correlation.correlation(R12,T12,t_left,t_right,cam_mats['t_left'],cam_mats['t_right'],u1,v1,window_size,Z)
                (correlation13,u2max_downl,v2max_downl) = correlation.correlation(R13,T13,t_left,b_left,cam_mats['t_left'],cam_mats['b_left'],u1,v1,window_size,Z)
                (correlation14,u2max_downr,v2max_downr) = correlation.correlation(R14,T14,t_left,b_right,cam_mats['t_left'],cam_mats['b_right'],u1,v1,window_size,Z)
                
                correlationAll = np.array([correlation12,correlation13,correlation14])
                correlationMean = np.mean(correlationAll,axis=0)
                correlationMean = correlationMean.tolist()
                Z_final = Z[correlationMean.index(max(correlationMean))]
                
                '''5mm'''
                Z = range(Z_final-5, Z_final+6, 5)
                (correlation12,u2max_right,v2max_right) = correlation.correlation(R12,T12,t_left,t_right,cam_mats['t_left'],cam_mats['t_right'],u1,v1,window_size,Z)
                (correlation13,u2max_downl,v2max_downl) = correlation.correlation(R13,T13,t_left,b_left,cam_mats['t_left'],cam_mats['b_left'],u1,v1,window_size,Z)
                (correlation14,u2max_downr,v2max_downr) = correlation.correlation(R14,T14,t_left,b_right,cam_mats['t_left'],cam_mats['b_right'],u1,v1,window_size,Z)
                
                correlationAll = np.array([correlation12,correlation13,correlation14])
                correlationMean = np.mean(correlationAll,axis=0)
                correlationMean = correlationMean.tolist()
                Z_final = Z[correlationMean.index(max(correlationMean))]
        
        Zmap.append(Z_final)
        print('the Z_final for pixel (%d,%d) is %d') %(v1,u1,Z_final)
        
ZmapSizeX = (480-2*window_size)/pixel_step
ZmapSizeY = (720-2*window_size)/pixel_step
Zmap = np.reshape(Zmap,((480-2*window_size)/pixel_step,(720-2*window_size)/pixel_step))




print "generating ply file"

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num = len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

imgL_color = t_left
colors = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB) 
out_points = np.zeros([ZmapSizeX*ZmapSizeY,3])
out_colors = np.zeros([ZmapSizeX*ZmapSizeY,3])


u0_t_left = cam_mats['t_left'][0][2]
v0_t_left = cam_mats['t_left'][1][2]
f = cam_mats["t_left"][0,0]
for i in range(0,ZmapSizeX):
    for j in range(0,ZmapSizeY):
        Z_ply = Zmap[i][j]
        u1 = j*pixel_step + window_size
        v1 = i*pixel_step + window_size
        X_ply = Z_ply*(u1-u0_t_left)/f
        Y_ply = Z_ply*(v1-v0_t_left)/f
        out_points[i*ZmapSizeY+j][0] = X_ply
        out_points[i*ZmapSizeY+j][1] = Y_ply
        out_points[i*ZmapSizeY+j][2] = Z_ply
        out_colors[i*ZmapSizeY+j][0] = colors[v1][u1][0]
        out_colors[i*ZmapSizeY+j][1] = colors[v1][u1][1]
        out_colors[i*ZmapSizeY+j][2] = colors[v1][u1][2]
     
        

write_ply(sample+'_correlation1234_dichotomy_threshold.ply', out_points, out_colors)

end = time.clock()
text_file = open(sample + "_correlation1234_dichotomy_threshold_running_time.txt", "w")
text_file.write("Running time: %s seconds" %(end-start))
text_file.close()
print('Running time: %s Seconds'%(end-start))