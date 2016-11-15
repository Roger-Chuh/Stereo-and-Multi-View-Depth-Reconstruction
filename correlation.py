# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:06:46 2016

@author: xumr
"""

import numpy as np
import cv2

class correlation:
    
    def correlation(self,R,T,img1,img2,cam_mats1,cam_mats2,u1,v1,window_size,Z):
        interpolation = cv2.INTER_LANCZOS4
        correlation = []
        f = cam_mats1[0,0]
    
        RT = np.concatenate((R,T),axis=1)
        window1 = img1[v1-window_size:v1+window_size+1,u1-window_size:u1+window_size+1]
        
        u0_1 = cam_mats1[0][2]
        v0_1 = cam_mats1[1][2]
    
        
        for z in Z: 
            X = z*(u1-u0_1)/f
            Y = z*(v1-v0_1)/f
            XYZ1 = np.array([[X],[Y],[z],[1]])
            XYZ_afterRT = np.dot(RT,XYZ1)
            u2v2s = np.dot(cam_mats2,XYZ_afterRT)
            s = u2v2s[2]
            u2v21 = u2v2s/s
    
            u2 = u2v21[0][0]
            v2 = u2v21[1][0]
    
            map1 = np.zeros((window_size*2+1,window_size*2+1))
            map2 = np.zeros((window_size*2+1,window_size*2+1))
            
            for i in range(0,window_size*2+1):
                for j in range(0,window_size*2+1):
                    map1[i][j] = u2-window_size+j
                    
            for i in range(0,window_size*2+1):
                for j in range(0,window_size*2+1):
                    map2[i][j] = v2-window_size+i
            
            map1 = np.float32(map1)
            map2 = np.float32(map2)    
            
            window2 = cv2.remap(img2,map1,map2,interpolation)
            
        
            
        #    window_right = right[v2-window_size:v2+window_size,u2-window_size:u2+window_size]
            
            if np.shape(window1)==np.shape(window2):
                res = cv2.matchTemplate(window1,window2,cv2.TM_CCOEFF_NORMED)
                correlation.append(res[0][0])
         
                if res[0][0] == max(correlation):
                    u2max = u2
                    v2max = v2
            else:
                correlation.append(-1)
    
        return correlation,u2max,v2max