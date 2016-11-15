# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 07:13:21 2016

@author: xumr
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from calibration import calibration
import time

start =time.clock()



'''Initialization'''

sample = 'Sample1'

calibration = calibration()
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = {"t_left": [], "t_right": []} # 2d points in image plane.
image_size = (720L, 480L)
#: Camera matrices (M)
cam_mats = {"t_left": None, "t_right": None}
#: Distortion coefficients (D)
dist_coefs = {"t_left": None, "t_right": None}
rvecs = {"t_left": None, "t_right": None} #from camer to image?
tvecs = {"t_left": None, "t_right": None}
sides = ['t_left','t_right']
rect_trans = {"t_left": None, "t_right": None}
proj_mats = {"t_left": None, "t_right": None}
valid_boxes = {"t_left": None, "t_right": None}
map1 = {"t_left": None, "t_right": None}
map2 = {"t_left": None, "t_right": None}


'''Camera Calibration'''

for side in sides:
    (objpoints, imgpoints[side], cam_mats[side],dist_coefs[side], rvecs[side],tvecs[side]) = \
        calibration.CameraCalibration(side)



'''Stereo Calibration'''

print "Doing Stereo Calibration for camera 12"
(R12,T12) = calibration.StereoCalibration(objpoints,imgpoints['t_left'],imgpoints['t_right'],
                                            cam_mats['t_left'],cam_mats['t_right'],dist_coefs['t_left'],dist_coefs['t_right'])

                              




    
'''Stereo Rectification'''
    
print "Doing Stereo Rectification..."

(rect_trans["t_left"], rect_trans["t_right"],
 proj_mats["t_left"], proj_mats["t_right"],
 disp_to_depth_mat, valid_boxes["t_left"],
 valid_boxes["t_right"]) = cv2.stereoRectify(cam_mats["t_left"],
                                           dist_coefs["t_left"],
                                           cam_mats["t_right"],
                                           dist_coefs["t_right"],
                                           image_size,
                                           R12,
                                           T12,
                                           flags=cv2.CALIB_ZERO_DISPARITY,
                                           alpha=1)

print "Getting the maps for undistortion and rectification..."



for side in ("t_left", "t_right"):
    
# (newcameramtx[side], roi[side]) = cv2.getOptimalNewCameraMatrix(cam_mats[side],dist_coefs[side],image_size,0)
    
 (map1[side],
 map2[side]) = cv2.initUndistortRectifyMap(
                                   cam_mats[side],
                                   dist_coefs[side],
                                   rect_trans[side],
                                   proj_mats[side],      #I used this before. probably wrong
#                                   cam_mats[side],    #for testing
#                                   cam_mats[side],
                                   image_size,
                                   cv2.CV_32FC1)

'''Read Images'''
t_left_fn = sample + '_t_left.jpeg'
t_right_fn = sample + '_t_right.jpeg'
t_left = cv2.imread(t_left_fn)
t_right = cv2.imread(t_right_fn)

print "Rectifying&Undistorting the two images "
t_left_UndRec = cv2.remap(t_left,map1["t_left"],map2["t_left"],cv2.INTER_NEAREST)
t_right_UndRec = cv2.remap(t_right,map1["t_right"],map2["t_right"],cv2.INTER_NEAREST)

cv2.imwrite(sample + '_t_left_UndRec.jpg',t_left_UndRec)
cv2.imwrite(sample + '_t_right_UndRec.jpg',t_right_UndRec)

msg = '''I need to save the images before drawing lines on them!'''
imgL = cv2.imread(sample + '_t_left_UndRec.jpg')
imgR = cv2.imread(sample + '_t_right_UndRec.jpg')

print "drawing lines on the result..."

cv2.line(t_left_UndRec,(0,200),(720,200),(0,0,255))
cv2.line(t_right_UndRec,(0,200),(720,200),(0,0,255))
cv2.line(t_left_UndRec,(0,60),(720,60),(0,0,255))
cv2.line(t_right_UndRec,(0,60),(720,60),(0,0,255))
cv2.line(t_left_UndRec,(0,120),(720,120),(0,0,255))
cv2.line(t_right_UndRec,(0,120),(720,120),(0,0,255))
cv2.line(t_left_UndRec,(0,240),(720,240),(0,0,255))
cv2.line(t_right_UndRec,(0,240),(720,240),(0,0,255))
cv2.line(t_left_UndRec,(0,300),(720,300),(0,0,255))
cv2.line(t_right_UndRec,(0,300),(720,300),(0,0,255))
cv2.line(t_left_UndRec,(0,360),(720,360),(0,0,255))
cv2.line(t_right_UndRec,(0,360),(720,360),(0,0,255))
cv2.line(t_left_UndRec,(0,420),(720,420),(0,0,255))
cv2.line(t_right_UndRec,(0,420),(720,420),(0,0,255))
cv2.line(t_left_UndRec,(0,10),(720,10),(0,0,255))
cv2.line(t_right_UndRec,(0,10),(720,10),(0,0,255))

cv2.imwrite(sample + '_t_left_UndRec_red_line.jpg',t_left_UndRec)
cv2.imwrite(sample + '_t_right_UndRec_red_line.jpg',t_right_UndRec)

cv2.imshow('hh',imgL)
cv2.waitKey(500)

#cv2.imshow('aa',a)
#cv2.waitKey(500)

cv2.imshow('left',t_left_UndRec)
cv2.waitKey(500)
cv2.imshow('right',t_right_UndRec)
cv2.waitKey(500)
cv2.destroyAllWindows()

#cv2.imwrite('l_UndRec.jpg',l_UndRec)
#cv2.imwrite('r_UndRec.jpg',r_UndRec)


msg = '''Priviously I used the images with red line on them OMG!!!'''

#imgL = cv2.imread('l_UndRec.jpg',0)
#imgR = cv2.imread('r_UndRec.jpg',0)

#stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=32,SADWindowSize=15)
#
#disparity = stereo.compute(imgL,imgR)
# disparity range is tuned for 'aloe' image pair
window_size = 8
min_disp = 0
num_disp = 112-min_disp
stereo = cv2.StereoSGBM(minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)

print 'Computing disparity...'
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0


print(np.max(disparity))
plot = plt.imshow(disparity,'gray')
plt.axis('off')
savefig_fn = sample + '_disparity_map.png'
plt.savefig(savefig_fn, bbox_inches='tight')
plt.show()

#generate 3d point cloud
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


h, w = imgL.shape[:2]
f = 0.8*w
Q = disp_to_depth_mat #also called perspective transformation matrix
points = cv2.reprojectImageTo3D(disparity,Q)
#colors = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
imgL_color = imgL  #cv2.imread('l_UndRec.jpg')
colors = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)
#mask = disparity > disparity.min()
#mask = (points > points.min()) * (points < points.max())
#mask = np.abs(points) < points.max()
mask = (disparity>0)  #(disparity > -10000) * (disparity < 10000)
out_points = points[mask]
out_colors = colors[mask]
out_fn = sample + '_stereo.ply'
write_ply(out_fn, out_points, out_colors)

print "done"




'''Print and save the running time'''

end = time.clock()

text_file = open(sample + "_running_time.txt", "w")
text_file.write("Running time: %s seconds" %(end-start))
text_file.close()

print('Running time: %s Seconds'%(end-start))
