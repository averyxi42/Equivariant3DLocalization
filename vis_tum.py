import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

import project

data = project.load_data()

row = data[0]

depth_name = row[2]
rgb_name = row[3]
color_raw = o3d.io.read_image(
    "data/rgbd_dataset_freiburg1_xyz/"+rgb_name)
depth_raw = o3d.io.read_image(
    "data/rgbd_dataset_freiburg1_xyz/"+depth_name)
print("Read TUM dataset")
rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
    color_raw, depth_raw)
print(rgbd_image)
plt.subplot(1, 3, 1)
plt.title('TUM grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 3, 2)
plt.title('TUM depth image')
plt.imshow(rgbd_image.depth)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
# Flip it, otherwise the pointcloud will be upside down

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#o3d.visualization.draw_geometries([pcd])c
colors = np.array(color_raw).reshape(-1,3)

depths = np.array(depth_raw).reshape(-1)

indices = depths>0

colors_valid = colors[indices]

colors_valid.shape

import torch
indices = torch.randperm(len(pcd.points))[:40960]

points = np.asarray(pcd.points)[indices]
colors_valid = colors_valid[indices]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
ax.scatter3D(points[:,0],points[:,1],points[:,2],s=0.01,c=colors_valid/255)
ax.view_init(elev=0, azim=60)
plt.show()