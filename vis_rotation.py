import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
gt = np.loadtxt('results/'+'gt_test.txt', delimiter=',')
estimates = np.loadtxt('results/'+'estimates_test.txt', delimiter=',')

print(gt)
print(estimates.shape)
print(estimates.dtype)

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
plt.plot(x,est_euler[:,2])
plt.plot(x,gt_euler[:,2])
plt.show()