#!/bin/bash
#This file sets up dependencies for the project. not tested on many computers, use at your own risk
cwd=$(pwd)
sudo apt-get update
sudo apt-get install -y python3.8-full
sudo apt-get install -y python3.8-dev
sudo apt-get install -y python3.8-distutils
sudo apt-get install -y python3.8-venv

sudo apt-get install -y pkg-config
sudo apt-get install -y libcairo2
sudo apt-get install -y python3-pybind11
python3.8 -m venv venv
source venv/bin/activate 
python -m pip install -r requirements.txt
cd se3_equivariant_place_recognition/vgtk
sudo $cwd/venv/bin/python3 setup.py build_ext --inplace
python -m ipykernel install --name python3.8 --user
cd $cwd