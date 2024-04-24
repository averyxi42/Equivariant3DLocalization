Hi thanks for reading this.
This repository contains the code for Robust 3D Localization with SE3 Equivariant Point Cloud Network.
[Our report](https://drive.google.com/file/d/1LalLpp9W9WZYnYGNh0vhaI1mzrRvhlry/view?usp=sharing)
## Abstract
 In this project, we developed a point-cloud localization method that employs equivariant convolutional layers and a particle filter. By inputting point cloud data into a pre-trained E2PN Generalized Mean model, we obtained SE(3)-equivariant features that exhibit robustness against spatial transformations. We devised a sensor model that utilizing these features and implemented a particle filter to estimate 3D pose trajectory on three sequences from the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset). We additionally compared our algorithm against a baseline using Scan Context. Our experiments on the TUM dataset demonstrated that our algorithm automatically injects point clouds with ground truth without need for manual landmark annotation, in addition to maintaining robust data association for localization.
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
## E2PN pre-trained model
In our project, we used pretrained E2PN model for extracting equivariant features from point clouds. The code for getting E2PN is:
```
git checkout E2PN
```

depending on your setup, you may also need to do *remeber to select python3.8 kernel when running test.ipynb*
```
pip install progress
```
We collect our testing result in the results folder. Some timestamp alignment is required to run the results in the [online toolbox](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/online_evaluation)of TUM dataset.
## Reference Code
- [E2PN](https://github.com/minghanz/EPN_PointCloud): Efficient SE(3)-Equivariant Point Network used in SE(3)-Equivariant Point Cloud-based Place Recognition as local feature extractor.
- [SE(3)-Equivariant Point Cloud-based Place Recognition](https://github.com/UMich-CURLY/se3_equivariant_place_recognition) We used the modified E2PN version from this repo and used it as SE(3)-invariant point cloud local feature extractor.



