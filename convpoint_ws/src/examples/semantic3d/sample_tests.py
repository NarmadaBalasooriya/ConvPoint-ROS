import numpy as np 
import open3d
from preprocess import point_cloud_txt_to_pcd

points_path = "./data/sg27_station1_intensity_rgb.pcd"

pcd = open3d.io.read_point_cloud(points_path)
pts = np.asarray(pcd.points)
clr = np.asarray(pcd.colors)

#arr = np.concatenate((pts, clr), axis=1)

point_cloud_txt_to_pcd('train_sample', 'bildstein_station3_xyz_intensity_rgb')
