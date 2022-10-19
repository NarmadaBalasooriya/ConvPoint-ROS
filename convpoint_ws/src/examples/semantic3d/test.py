# Semantic3D Example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')

import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import open3d


import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, accuracy_score
import time
import utils.metrics as metrics
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from PIL import Image

from point_cloud_util import load_labels

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

# dataset for test
class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=8,
                    npoints = 8192,
                    test_step=0.8, nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = nocolor
        self.filename = filename

        # load the points
        pcd = open3d.io.read_point_cloud(os.path.join(self.folder, self.filename))
        pts = np.asarray(pcd.points)
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

def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)


def eval_hyperparameters(gt_labels, pr_labels):
    accuracy_value = accuracy_score(gt_labels, pr_labels)
    return accuracy_value

def test(batch_size, no_points):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', '-s', help='Path to data folder')
    parser.add_argument("--savedir", type=str, default="./pretrained_model/SegBig_nocolor/current/SegBig_8192_nocolorTrue_drop0.5_2021-02-27-02-05-06/")
    parser.add_argument('--block_size', help='Block size', type=float, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iter", "-i", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", default=8.0, type=float)
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--drop", default=0.5, type=float)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_{}_nocolor{}_drop{}_{}".format(
            args.model, args.npoints, args.nocolor, args.drop, time_string))


    filelist_test = [
        #"Paradise_cropped_0.5.pcd"
        #"holyrood_upper.pcd"
        #"Hollyrood_no_crop.pcd"
        "test1.pcd"
        ]

    N_CLASSES = 2
    old_n_classes = 8

    
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
        net.load_state_dict(torch.load(os.path.join(args.savedir, "semantic3d_landing_state_dict.pth")))
    else:
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
        net.load_state_dict(torch.load(os.path.join(args.savedir, "semantic3d_landing_state_dict.pth")))
        
    net.cuda()
    print("Done: ", next(net.parameters()).device)


    ##### TRAIN
    
    net.eval()
    total_runtime = []
    for filename in filelist_test:
        print(filename)
        ds = PartDatasetTest(filename, args.rootdir,
                        block_size=args.block_size,
                        npoints= no_points,
                        test_step=args.test_step,
                        nocolor=args.nocolor
                        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                        num_workers=args.threads
                                        )

        xyzrgb = ds.xyzrgb[:,:3]
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        start = time.time()
        with torch.no_grad():
            t = tqdm(loader, ncols=100)
            print(len(t), len(loader))
            for pts, features, indices in t:                
                features = features.cuda()
                pts = pts.cuda()
                
                start_pr = time.time() # start the prediction time
                outputs = net(features, pts)
                end_pr = time.time() # end the prediction time
                
                pr_time = end_pr - start_pr
                
                start_post = time.time() # prediction post processing

                outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                scores[indices.cpu().numpy().ravel()] += outputs_np
                
                end_post = time.time() # end time prediction post processing
                
                post_time = end_post - start_post # post processing time taken                
                pr_t = f"{pr_time:.1f}"
                pt_t = f"{post_time:.1f}"
                
                t.set_postfix(PR_Time=pr_t, Post_Time=pt_t)
                

        mask = np.logical_not(scores.sum(1)==0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

        # compute softmax
        scores = scores - scores.max(axis=1)[:,None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
        scores = np.nan_to_num(scores)

        end = time.time()

        #os.makedirs(os.path.join(args.savedir, "results"), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, "results", f"{filename}.txt")
        scores = scores.argmax(1)
        #np.savetxt(save_fname,scores,fmt='%d')

        # if args.savepts:
        #     save_fname = os.path.join(args.savedir, "results", f"{filename}_pts.txt")
        #     xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
        #     np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])

        # break
        print('Eval time: ', end-start)
        total_runtime.append(end-start)
        
    

