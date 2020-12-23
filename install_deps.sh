#!/bin/bash

CLEAR_ENV=$1
if [ -z $CLEAR_ENV ]
then
	echo "Pass in true (or any word) as the first command line argument to remake the virtual environment from scratch"
	sleep 1
fi

if [ -z $NOT_NVIDIA ]
then
	echo Type \"NOT_NVIDIA=true $0\" to install if you are using your desktop instead of a Jetson device
	sleep 1
fi

VENV_DIR=env

sudo apt-get update
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev ffmpeg
sudo apt-get install -y python3-venv python3-pip


if [ $CLEAR_ENV ] && [ -e $VENV_DIR ]
then
	rm -rf $VENV_DIR
	echo "Deleted previous virtual environment"
fi
echo "Creating Python3 virtual environment $VENV_DIR"
python3 -m venv $VENV_DIR

source $VENV_DIR/bin/activate
pip3 install --upgrade pip testresources setuptools
echo "Installing numpy and other packages (will take about half an hour)"
pip3 install --upgrade numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf


if [ $NOT_NVIDIA ]
then
	pip3 install tensorflow==1.15.0
else
	pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu==1.15.0+nv20.1
fi


echo Installed TensorFlow.

# INSTALLING TF OBJECT DETECTION API
# Based on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

sudo apt-get install -y protobuf-compiler
sudo apt-get install -y libxml2-dev libxslt1-dev # Necessary to install lxml
sudo apt-get install -y libfreetype6-dev         # Necessary for matplotlib
echo "Installing lxml (this will take about 15 minutes)"
pip3 install lxml
pip3 install Cython
pip3 install contextlib2
pip3 install pillow
echo "Installing matplotlib (this will take about 10 minutes)"
pip3 install matplotlib

# Based upon new updates from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md
pip3 install pandas==1.1.1
pip3 install pytz==2020.1
pip3 install scipy==1.5.2
pip3 install tf-slim==1.1.0

REPO_DIR=~/git/tf_models/
PREV_DIR=`pwd`
mkdir -p $REPO_DIR
git clone https://github.com/tensorflow/models.git $REPO_DIR
cd $REPO_DIR/research
protoc object_detection/protos/*.proto --python_out=.
pip3 install .
cd $PREV_DIR

pip3 install pycocotools

deactivate
echo Installation finished.
echo Run \"source $VENV_DIR/bin/activate\" to enter the virtual environment with TensorFlow.
echo Note that you will not be able to access Jetson Inference from said virtual environment.
