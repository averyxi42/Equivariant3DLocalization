import numpy as np

# Read data from text file
# gt = np.loadtxt('results/'+'gt_test.txt', delimiter=',')
# estimates = np.loadtxt('results/'+'estimates_test.txt', delimiter=',')
gt = np.loadtxt('results/'+'gt_desk.txt', delimiter=',')
estimates = np.loadtxt('results/'+'estimates_desk.txt', delimiter=',')

print(gt)
print(estimates.shape)
print(estimates.dtype)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming estimates is the trajectory array with shape (2654, 7)
estimates_positions = estimates[:, :3]  # Extract x, y, z positions from estimates
gt_positions = gt[:, :3] 

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize empty plot
gt_line, = ax.plot([], [], [], color='blue', lw=2, label='Ground Truth')
estimates_line, = ax.plot([], [], [], color='red', lw=2, label='Estimates')

# Set plot limits
# Set plot limits
ax.set_xlim3d([np.min(estimates_positions[:, 0]), np.max(estimates_positions[:, 0])])
ax.set_ylim3d([np.min(estimates_positions[:, 1]), np.max(estimates_positions[:, 1])])
ax.set_zlim3d([np.min(estimates_positions[:, 2]), np.max(estimates_positions[:, 2])])

# Update function for animation
def update(frame):
    gt_line.set_data(gt_positions[:frame, 0], gt_positions[:frame, 1])
    gt_line.set_3d_properties(gt_positions[:frame, 2])
    
    estimates_line.set_data(estimates_positions[:frame, 0], estimates_positions[:frame, 1])
    estimates_line.set_3d_properties(estimates_positions[:frame, 2])
    
    return gt_line, estimates_line

# Create animation object
ani = FuncAnimation(fig, update, frames=len(gt_positions), interval=0.02)

# Show the animation
plt.legend()
plt.show()