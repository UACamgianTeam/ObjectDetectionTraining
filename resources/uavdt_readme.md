1. Data preparation
  (1). Put all the images into /UAV2017/JPEGImages
  (2). Convert all groundtruth files to XML format, and put into /UAV2017/Annotations_2classes (for 2 classes) or /UAV2017/Annotations_4classes (for 4 classes)
  (3). Train and test lists are at /UAV2017/ImageSets/Layout

2. SSD
   (1). export PYTHONPATH=/home/kwduan/UAV_Det_codes/ssd_car_2classes/python:$PYTHONPATH
   (2). make all -j16
   (3). make pycaffe
   (4). cd ssd_car_2classes/data/uav/
   (5). ./create_list.sh
   (6). ./create_data.sh
   (7). cd ssd_car_2classes/examples/ssd/models/VGGNet/UAV/SSD_300x300, copy all the contents in VGG_UAV_SSD_300x300.sh
   (8). cd ssd_car_2classes/, paste

3. RON
   (1). export PYTHONPATH=/home/kwduan/UAV_Det_codes/RON_car_2classes/caffe-ron/python:$PYTHONPATH
   (2). cd RON_car_2classes/caffe-ron/
   (3). make all -j16
   (4). make pycaffe
   (5). cd RON_car_2classes/lib/
   (6). make
   (7). cd RON_car_2classes/data/
   (8). ln -s $VOCdevkit VOCdevkit2007
   (9). ln -s /home/kwduan/UAV_Det_codes/UAV2017 VOCdevkit2007
   (10). cd RON_car_2classes/
   (11). ./train_uav.sh
   (12). ./test_uav.sh

4. Faster-rcnn
   (1). export PYTHONPATH=/home/kwduan/UAV_Det_codes/faster-rcnn_2classes/caffe-fast-rcnn/python:$PYTHONPATH
   (2). cd faster-rcnn_2classes/caffe-fast-rcnn/
   (3). make all -j16
   (4). make pycaffe
   (5). cd faster-rcnn_2classes/lib/
   (6). make
   (7). cd faster-rcnn_2classes/data/
   (8). ln -s $VOCdevkit VOCdevkit2007
   (9). ln -s /home/kwduan/UAV_Det_codes/UAV2017 VOCdevkit2007
   (10). cd faster-rcnn_2classes/
   (11). ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 uav2017

5. R-FCN
   (1). export PYTHONPATH=/home/kwduan/UAV_Det_codes/R-FCN_2classes/caffe/python:$PYTHONPATH
   (2). cd R-FCN_2classes/caffe/
   (3). make all -j16
   (4). make pycaffe
   (5). cd R-FCN_2classes/lib/
   (6). make
   (7). cd faster-rcnn_2classes/data/
   (8). ln -s $VOCdevkit VOCdevkit2007
   (9). ln -s /home/kwduan/UAV_Det_codes/UAV2017 VOCdevkit2007
   (10). cd R-FCN_2classes/
   (11). export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
   (12). export CUDA_HOME=/usr/local/cuda
   (13). ./experiments/scripts/rfcn_end2end.sh 0 ResNet-50 uav2017
   (14). for test only: ./experiments/scripts/test_only.sh 0 ResNet-50 uav2017


