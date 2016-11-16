# Stereo-and-Multi-View-Depth-Reconstruction
This project compares the result of Stereo Rectification and Multi-View Depth Reconstruction by comparing the 3D Point Clouds genereated by this two methods.
calibration.py contains classes of camera calibration and stereo calibration.
correlation.py contains the methods of computing depth value by comparing window correlation. It is for the Muli-View Depth Reconstruction.
correlation1234_ply_new_dichotomy_threshold.py is the code of Multi-View Depth Reconstruction algorithm.
stereo reconstruction.py is the code of Stereo Reconstruction.
calibration images folder contains all the images used for camera calibration and stereo calibration in this project. You need to copy them to the same direction of all the code before running the code.
