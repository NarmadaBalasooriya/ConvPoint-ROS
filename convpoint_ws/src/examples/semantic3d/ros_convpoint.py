import sys
from gpg import Data
sys.path.append('../../')

import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import open3d
import csv

import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import time
import utils.metrics as metrics
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from PIL import Image

from point_cloud_util import load_labels

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

print('-----------------------------------')

print('GPU device: ', torch.cuda.get_device_name(device))
print('------------------------------------')

start = 0
end = 0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    print(indices.shape)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)

def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)

class DatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, pcd_np,
                    block_size,
                    npoints,
                    test_step=0.8, nocolor=False):

        
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = True

        # load the points
        
        pts = pcd_np
        #clrs = np.asarray(pcd.colors)
        clrs = np.zeros((pts.shape))

        self.xyzrgb = np.concatenate((pts, clrs), axis=1)
        #self.xyzrgb = np.load(os.path.join(self.folder, self.filename))
        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points
        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:,3:6]
            fts = fts.astype(np.float32)
            fts = fts / 255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return len(self.pts)

class RosConvPoint():
	def __init__(self, modeldir, logdir, batch_size, npoints, test_step=0.8, nocolor=False):
     	
		self.modeldir = modeldir
		self.logdir = logdir
		self.batch_size = batch_size
		self.bs = 16
		self.npoints = npoints
		self.verbose = False
		self.nocolor = nocolor
		
		self.threads = 4
		self.n_classes = 2
		self.old_n_classes = 8
		self.model_name = 'SegBig'
  
		## Model
		self.net = get_model(self.model_name, input_channels=1, output_channels=self.n_classes, args=args)
		self.net.load_state_dict(torch.load(os.path.join(self.modeldir, "semantic3d_landing_state_dict.pth")))
		
  		## Time log file
		self.time_file = os.path.join(self.logdir, "infer_time_ros.csv")
		self.header = {"msg no", "no_points", "time"}
  
		# save pcd
		os.makedirs(os.path.join(self.logdir, "predictions"))

		## net to cuda 
		self.net.cuda()

		self.net.eval()
		## ros subscriber and publisher
		self.seg_pub = rospy.Publisher("segmented_supmap_pcl", PointCloud2, queue_size=1)
		self.vel_sub = rospy.Subscriber("submap_pcl", PointCloud2, self.infer_callback, queue_size=1)

		rospy.loginfo("Initialization done. Running inference")

	def infer_callback(self, msg):
		
		rospy.loginfo("Read pointcloud from ros subscriber")
		points = point_cloud2.read_points(msg)
		self.velodyne_points = np.asarray(list(points), dtype=np.float32)
		ds = DatasetTest(self.velodyne_points, self.bs, self.npoints)
		loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=self.threads)
		xyzrgb = ds.xyzrgb[:,:3]
		scores = np.zeros((xyzrgb.shape[0], self.n_classes))

		with open(self.time_file, "w", newline="") as csvfile:

			times = csv.writer(csvfile)
			times.writerow(self.header)

			start_t = time.time()
			i = 0
   
			with torch.no_grad():
				t = tqdm(loader, ncols=100)
				for pts, features, indices in t:
					features = features.cuda()
					pts = pts.cuda()
					outputs = self.net(features, pts)
					outputs_np = outputs.cpu().numpy().reshape((-1, self.n_classes))
					scores[indices.cpu().numpy().ravel()] += outputs_np

			mask = np.logical_not(scores.sum(1)==0)
			scores = scores[mask]
			pts_src = xyzrgb[mask]
			scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)
   
			scores = scores - scores.max(axis=1)[:,None]
			scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
			scores = np.nan_to_sum(scores)

			save_fname = os.path.join(self.logdir, "predictions", str(i)+"_lbls.txt")
			xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
			np.savetxt(save_fname, xyzrgb, fmt=['%.4f','%.4f','%.4f','%d'])

			end_t = time.time()
			rospy.loginfo("Infered msg ", str(i), " in ", str(end_t-start_t), "sec")
			times.writerow((i, len(self.velodyne_points), (end_t-start_t)))
			i+=1
			return

	
	

if __name__ == '__main__':
	rospy.init_node("convpoint_ros", anonymous=True)
	rospy.loginfo("Initializing rosnode")

	arch_file = rospy.get_param("~arch_file")
	data_file = rospy.get_param("~data_file")
	model_dir = rospy.get_param("~model_dir")
	log_dir = rospy.get_param("~log_file")
	batch_size = rospy.get_param("~batch_size")
	npoints = rospy.get_param("~npoints")
	rospy.loginfo("config and model files loaded")

	try:
		rospy.loginfo("Running RosConvPoint")
		seg_obj = RosConvPoint(model_dir, log_dir, batch_size, npoints, test_step=0.8, nocolor=False)
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting Down Node!!!")
		sys.exit(0)


