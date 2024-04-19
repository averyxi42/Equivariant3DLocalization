import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
gt = np.loadtxt('results/'+'gt_test_desk.txt', delimiter=',')
estimates = np.loadtxt('results/'+'estimates_test_desk.txt', delimiter=',')

print(gt)
print(estimates.shape)
print(estimates.dtype)
plt.rcParams.update({'font.size': 20})
gt_quat = gt[:,3:]#[3:]
est_quat = estimates[:,3:]
gt_euler = []
est_euler = []
for quat in gt_quat:
    gt_euler.append(Rotation.from_quat(quat).as_euler("xyz"))

gt_euler = np.stack(gt_euler)

for quat in est_quat:
    est_euler.append(Rotation.from_quat(quat).as_euler("xyz"))

est_euler = np.stack(est_euler)

x = np.arange(len(est_euler))
fig = plt.figure()
ax = fig.add_subplot(311)
ay = fig.add_subplot(312)
az = fig.add_subplot(313)

i = np.arange(len(gt_euler))
ax.plot(i,gt_euler[:,0],label="ground truth")
ax.plot(i,est_euler[:,0],label = "estimate")
ax.legend()
ax.set_title("roll")
ax.set_xticklabels([])
ay.plot(i,gt_euler[:,1],label="ground truth")
ay.plot(i,est_euler[:,1],label = "estimate")
ay.legend()
ay.set_title("pitch")
ay.set_xticklabels([])
az.plot(i,gt_euler[:,2],label = "ground truth")
az.plot(i,est_euler[:,2],label="estimate")
az.legend()
az.set_title("yaw")
az.set_xticklabels([])
plt.show()