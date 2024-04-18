import sys
import os
sys.path.append('se3_equivariant_place_recognition/vgtk' )

#from SPConvNets.options import opt as opt_oxford
import progress.bar
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import se3_equivariant_place_recognition.SPConvNets.models.e2pn_gem
import argparse
def load_model():
    opt = argparse.Namespace(
        device = 'cuda',
        model = argparse.Namespace(
            dropout_rate = 0,
            flag = 'attention',
            search_radius = 0.4,
            kpconv = None,
            kanchor = 12
        ),
        train_loss = argparse.Namespace(
            temperature = 3
        )
    )
    model = E2PNGeM(opt)
    checkpoint = torch.load('e2pn_gem_model.ckpt')
    saved_state_dict = checkpoint['state_dict'] 
    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)
    return model
def load_e2pn():
    opt = argparse.Namespace(
        device = 'cuda',
        model = argparse.Namespace(
            dropout_rate = 0,
            flag = 'attention',
            search_radius = 0.4,
            kpconv = None,
            kanchor = 12
        ),
        train_loss = argparse.Namespace(
            temperature = 3
        )
    )
    model = E2PNGeM(opt)
    checkpoint = torch.load('e2pn_gem_model.ckpt')
    saved_state_dict = checkpoint['state_dict'] 
    model.load_state_dict(saved_state_dict)
    return model.e2pn
def get_global_descriptor(model, network_input):
    with torch.no_grad():    

        # get output features from the model
        model = model.eval()
        network_output, _ = model(network_input)

        # tensor to numpy
        network_output = network_output.detach().cpu().numpy().squeeze()
        network_output = network_output.astype(np.double)
    
    return network_output


import numpy as np

def read_gt(file_path):
    data = np.genfromtxt(file_path, skip_header=3)  # Skip the first 3 lines
    return data[:, 1:8]  # Extract columns 1 to 3 (ax, ay, az)

# Function to read depth and color image filenames from file
def read_filenames(file_path):
    data = np.genfromtxt(file_path, skip_header=3, dtype=str)  # Skip the first 3 lines
    return data[:, 1]  # Extract column 1 (filename)
def read_timestamp(file_path):
    data = np.genfromtxt(file_path, skip_header=3)  # Skip the first 3 lines
    timestamps = data[:, 0]
    return timestamps

#load pcd from files
def load_pcd(dsname,depthname,rgbname):
    color_raw = o3d.io.read_image(
        dsname+"/"+rgbname)
    depth_raw = o3d.io.read_image(
        dsname+"/"+depthname)
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        color_raw, depth_raw)

    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

def load_data(name=None):
    if(name==None):
        name = "rgbd_dataset_freiburg1_xyz"

    # Read timestamps from files
    gt_timestamps = read_timestamp(name+ "/groundtruth.txt")
    depth_timestamps = read_timestamp(name+"/depth.txt")
    color_timestamps = read_timestamp(name+"/rgb.txt")

    gt = read_gt(name+"/groundtruth.txt")
    depth_filenames = read_filenames(name+"/depth.txt")
    color_filenames = read_filenames(name+"/rgb.txt")

    # Define time interval
    time_interval = 0.05

    # Initialize arrays to store synchronized data
    sync_data = []

    # Iterate through sensor timestamps
    i = 0
    for sensor_timestamp in gt_timestamps:
        # Find nearest depth and color timestamps within time interval
        closest_depth_idx = np.argmin(np.abs(depth_timestamps - sensor_timestamp))
        closest_color_idx = np.argmin(np.abs(color_timestamps - sensor_timestamp))
        
        # Check if time difference is within time interval
        if abs(depth_timestamps[closest_depth_idx] - sensor_timestamp) <= time_interval and \
        abs(color_timestamps[closest_color_idx] - sensor_timestamp) <= time_interval:
            
            # Store synchronized data
            sync_data.append([
                sensor_timestamp,
                gt[i],  # Append accelerometer data for corresponding depth image
                depth_filenames[closest_depth_idx],    # Append depth image filename
                color_filenames[closest_color_idx]     # Append color image filename
            ])
        i+=1

    # Convert synchronized data to array
    sync_data = np.array(sync_data)

    # Write synchronized data to a new file
    np.savetxt("sync.txt", sync_data, delimiter=",", fmt='%s', header="Sensor Timestamp, Ax, Ay, Az, Depth Filename, Color Filename", comments="")
    return sync_data

def pcd_at(dsname,data,index):
    return load_pcd(dsname,data[index,2],data[index,3])

from scipy.spatial import KDTree
#the tree has two halves, second halve is mirrored for quat
#data dimension: 2N x 7
def make_tree(sync_data):
    posearr = np.stack(sync_data[:,1])
    posemir = posearr.copy()
    posemir[:,4:] = - posearr[:,4:]
    pose = np.concatenate((posearr,posemir))
    tree = KDTree(pose)
    return tree
import open3d as o3d
import progress

import csv
def write_poses(sync_data,pose_data,fname):
    with open(fname+".txt",mode='w') as csvfile:
        writer = csv.writer(csvfile,delimiter=' ')
        for time,row in zip(sync_data[:,0],pose_data):
            writer.writerow([time,row[0],row[1],row[2],row[3],row[4],row[5],row[6]])
##buggy
def gen_features(model,sync_data):
    features =[]
    bar = progress.bar.PixelBar(max=len(sync_data))
    bs = 16 #batch size
    batch = torch.zeros(bs,4096,3)
    bc = 0
    for row in sync_data:
        depth_name = row[2]
        rgb_name = row[3]
        color_raw = o3d.io.read_image(
            "rgbd_dataset_freiburg1_xyz/"+rgb_name)
        depth_raw = o3d.io.read_image(
            "rgbd_dataset_freiburg1_xyz/"+depth_name)
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
            color_raw, depth_raw)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        points = np.asarray(pcd.points,dtype=np.float32)
        num_pts = len(points)
        indices =np.arange(0,num_pts,num_pts//4096)[:4096]
        points = points[indices]

        if(bc==bs):
            # print(batch.shape)
            features.append(get_global_descriptor(model,batch))
            # print(features[-1].shape)
            bc=0
        batch[bc] = torch.from_numpy(points)
        
        bar.next()
        bc+=1
    bar.finish()
    return np.concatenate(features,axis=0)
#convert o3d pointcloud to torch tensor ready for inference
def convert_pointcloud(pcd):
    points = np.asarray(pcd.points)
    num_pts = len(points)
    indices =np.arange(0,num_pts,num_pts//4096)[:4096]
    points = points[indices]
    return torch.from_numpy(points.reshape((1,4096,3))).float()

def inference(model,pcd):
    return get_global_descriptor(model,convert_pointcloud(pcd))
#sequentially generate the features for each point cloud.
def gen_features_seq(model,sync_data,name):
    if(name==None):
        name = "rgbd_dataset_freiburg1_xyz"
    features =[]
    bar = progress.bar.PixelBar(max=len(sync_data))
    for row in sync_data:
        depth_name = row[2]
        rgb_name = row[3]
        color_raw = o3d.io.read_image(
            name+"/"+rgb_name)
        depth_raw = o3d.io.read_image(
            name+"/"+depth_name)
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
            color_raw, depth_raw)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        points = np.asarray(pcd.points)
        num_pts = len(points)
        indices =np.arange(0,num_pts,num_pts//4096)[:4096]
        points = points[indices]

        features.append(get_global_descriptor(model,torch.from_numpy(points.reshape((1,4096,3))).float()))
        bar.next()
    bar.finish()
    return np.stack(features,axis=0) #return N x 256 np array

#query taking advantage of quaternion symmetry (identifying antipodal points)
def query(tree:KDTree,query):
    n = len(tree.data)//2
    d,i = tree.query(query)
    if(i>=n):
        i-=n
    return d,i

def distance(f1,f2):
    return np.linalg.norm(f1-f2)


from numpy.random import multivariate_normal
#disperse particles according to gaussian random walk
def predict_particles(particles):
    tcov = 0.01 #0.1
    qcov = 0.005
    out = []
    for particle in particles:
        t = particle[:3]+ multivariate_normal(np.zeros(3),np.diag([tcov,tcov,tcov]))
        quat = particle[3:] +multivariate_normal(np.zeros(4),np.diag([qcov,qcov,qcov,qcov]))
        quat/= np.linalg.norm(quat)
        out.append(np.concatenate((t,quat)))
    return np.stack(out)

import scipy.stats as stats
def correct_particles(particles,model,pcd,tree,features):
    weights = np.empty(len(particles))
    feature = inference(model,pcd)
    ind=0
    for particle in particles:
        dst,index  = query(tree,particle)
        #print(dst)
        weights[ind] = stats.norm(0,0.8).pdf(distance(feature,features[index]))*stats.norm(0,0.2).pdf(np.linalg.norm(dst)) #0,2
        ind+=1
    weights /= sum(weights)
    # weights = np.concatenate(weights)
    # weights/= np.sum(weights)
    return weights

def resample_particles(particles, weights):
    new_samples = np.zeros_like(particles)
    new_weight = np.zeros_like(weights)
    W = np.cumsum(weights)
    r = np.random.rand(1) / len(particles)
    count = 0
    for j in range(len(particles)):
            u = r + j/len(particles)
            while u > W[count]:
                count += 1
            new_samples[j,:] = particles[count,:]
            new_weight[j] = 1 / len(particles)
    particles = new_samples
    weights = new_weight
    return particles,weights

def get_moments(particles,weights):
    mean = np.sum(particles*weights.reshape((-1,1)),axis=0)
    mats = []
    cov = np.zeros((7,7))
    for i in range(len(particles)):
        particle = particles[i]
        cov+=((particle-mean).reshape((-1,1)))*((particle-mean).reshape((1,-1)))*weights[i]
    return mean,cov

def gaussian_resample(particles,weights,n):
    mean,cov = get_moments(particles,weights)
    cov = cov[np.arange(7),np.arange(7)]
    particles = multivariate_normal(mean,np.diag(cov)/20,n)
    norms = np.linalg.norm(particles[:,3:],axis=1)
    particles[:,3:] = particles[:,3:]/norms.reshape((-1,1))
    return particles