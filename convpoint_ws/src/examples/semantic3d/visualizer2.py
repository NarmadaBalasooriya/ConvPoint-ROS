import numpy as np
import open3d
import pptk
from preprocess import point_cloud_txt_to_pcd
from point_cloud_util import load_labels, colorize_point_cloud

'''
points_path = "./data/prepared/test/pointcloud/stgallencathedral_station1_intensity_rgb_voxels.npy"
points_txt_path = "./data/prepared/test/pointcloud_txt/stgallencathedral_station1_intensity_rgb_voxels.txt"
ori_points_path = "./data/TEST/stgallencathedral_station1_intensity_rgb.txt"
labels_path = "./pretrained_model/SegBig_rgb/results/stgallencathedral_station1_intensity_rgb_voxels.npy_pts.txt"

#labels = np.loadtxt(labels_path)
#points = np.load(points_path)
#points_txt = np.loadtxt(points_txt_path)
#ori_points = np.loadtxt(ori_points_path)
#labels = np.fromfile(labels_path)

#point_cloud_txt_to_pcd('data/prepared/test/pointcloud_txt', 'stgallencathedral_station1_intensity_rgb_voxels')
'''

#pt_voxel_path = "./test/points/sg27_station1_intensity_rgb.pcd"
#pt_voxel_label_path = "./test/labels/sg27_station1_intensity_rgb.pcd.txt"

'''
pt_voxel_path = "./train_sample/bildstein_station3_xyz_intensity_rgb.pcd"
pt_voxel_label_path = "./train_sample/bildstein_station3_xyz_intensity_rgb.labels"
pt_pred_label_path = "./train_sample/bildstein_station3_xyz_intensity_rgb.pcd.txt"

pcd = open3d.io.read_point_cloud(pt_voxel_path)
labels = load_labels(pt_voxel_label_path)
pred_labels = load_labels(pt_pred_label_path)

#colorize_point_cloud(pcd, labels)
colorize_point_cloud(pcd, pred_labels)

open3d.visualization.draw_geometries([pcd])
'''

pcd =open3d.io.read_point_cloud('./data/islab/labeled_point_clouds/Semantic/castleblatten_station1_intensity_rgb_0.5.pcd')
pcd2 = open3d.io.read_point_cloud('./train_sample/bildstein_station3_xyz_intensity_rgb.pcd')
#labels = load_labels('./data/islab/labeled_point_clouds/Semantic/castleblatten_station1_intensity_rgb_0.5_labels.txt')

#colorize_point_cloud(pcd, labels)

#open3d.visualization.draw_geometries([pcd])
pts1 = np.asarray(pcd.points)
pts2 = np.asarray(pcd.colors)
pts3 = np.asarray(pcd2)


lbls = np.loadtxt('./data/islab/labeled_point_clouds/Semantic/castleblatten_station1_intensity_rgb_0.5_labels.txt')
print(pts1.shape)
print(pts2.shape)
print(pts3.shape)
print(lbls[0].dtype)
