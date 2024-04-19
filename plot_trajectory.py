
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

gt = np.loadtxt('results/'+'gt_test.txt', delimiter=',')
estimates = np.loadtxt('results/'+'estimates_test.txt', delimiter=',')
plt.rcParams.update({'font.size': 20})
# Assuming estimates is the trajectory array with shape (2654, 7)
estimates_positions = estimates[:, :3]  # Extract x, y, z positions from estimates
gt_positions = gt[:, :3] 

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(311)
ay = fig.add_subplot(312)
az = fig.add_subplot(313)

i = np.arange(len(gt_positions))
ax.plot(i,gt_positions[:,0],label="ground truth")
ax.plot(i,estimates_positions[:,0],label = "estimate")
ax.legend()
ax.set_title("x")
ax.set_xticklabels([])

ay.plot(i,gt_positions[:,1],label="ground truth")
ay.plot(i,estimates_positions[:,1],label = "estimate")
ay.legend()
ay.set_title("y")
ay.set_xticklabels([])

az.plot(i,gt_positions[:,2],label = "ground truth")
az.plot(i,estimates_positions[:,2],label="estimate")
az.legend()
az.set_title("z")
az.set_xticklabels([])
plt.show()