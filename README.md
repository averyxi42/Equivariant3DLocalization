Hi thanks for reading this.
This repository contains the code for Robust 3D Localization with SE3 Equivariant Point Cloud Network.
## Abstract
 In this project, we developed a localization method using state-to-the-art E2PN convolution layer. By putting in point cloud data into a pre-trained E2PN Generalized Mean model, we got SE(3)-equivariant global descriptors, which is robust in terms of spacial transformation. We utilized the particle filter to finish the localization task on TUM RGB-D dataset to show our method is automated, unnecessary to mark landmark, and has robust data association. 
## Environment Set Up
 To set up the enviornment, first git clone from SE(3)-Equivariant Point Cloud-based Place Recognition's repo, as it contains important pre-trained E2PN model for extracting point-cloud features. It saves time for you. After performing this cd into the se3_equivariant_place_recognition folder.
``` 
 git clone https://github.com/UMich-CURLY/se3_equivariant_place_recognition
```
Replace the requirement.txt and setup.bash with those from Equivariant3DLocalization repo.

Then do
```
source setup.bash

cd vgtk
```
*Note this environment requires python3.8, if your version is higher than this, run
```
sudo ../venv/bin/python setup.py build_ext -i
```
##E2PN pre-trained model
In our project, we used pretrained E2PN model for extracting equivariant features from point clouds. The code for getting E2PN is:
```
git checkout E2PN
```

depending on your setup, you may also need to do
```
pip install progress
```
## Reference Code
- [E2PN](https://github.com/minghanz/EPN_PointCloud): Efficient SE(3)-Equivariant Point Network used in SE(3)-Equivariant Point Cloud-based Place Recognition as local feature extractor.
- [SE(3)-Equivariant Point Cloud-based Place Recognition](https://github.com/UMich-CURLY/se3_equivariant_place_recognition) We used the modified E2PN version from this repo and used it as SE(3)-invariant point cloud local feature extractor.



