# PEViD-UHD

## Description
- [PEViD-UHD](https://www.epfl.ch/labs/mmspg/downloads/pevid-uhd/)  is a 4k dataset consisting of 26 4K UHD video 
sequences, each 13 seconds long, with a frame resolution of
3840 x 2160 pixels and captured at 30 fps using a Samsung Galaxy Note 3 smartphone. 
Video sequences in the dataset depict several typical surveillance scenarios: walking, exchanging bags, fighting, 
and stealing, which were shot in outdoor and indoor environments. 

- Participants appearing in the video have various 
gender and race, they are dressed differently and carry various personal items and accessories. Their silhouettes 
were manually annotated and the annotations are provided in XML format. All participants have read and signed a 
consent form, allowing free usage of these video sequences for research purposes.

## Setup
- Run [install_deps.sh]() to install the dependencies needed to manipulate the 
PEViD-UHD dataset
```
# From <your_path>/ObjectDetectionTraining
bash PEViD-UHD/install_pevid-uhd_deps.sh
```

## Extraction
Annotations in this dataset are stored in a proprietary format. We wish to convert them to Pascal-VOC XML 
annotations.

1. For every video (.mp4) and annotation file (.xgtf), such as ```Walking_day_outdoor_3_4K.xgtf```
and ```Walking_day_outdoor_3_original.mp4```, create a subdirectory inside of [data/PEViD-UHD](../data/PEViD-UHD)
specifically for that video, such as ```Walking_day_outdoor_3/```. 

2. Create new folders for image frames and Pascal-VOC XML annotations, respectively:
    ```
    # From <your_path>/ObjectDetectionTraining
    mkdir data/PEViD-UHD/<video_name>/frames
    mkdir data/PEViD-UHD/<video_name>/annotations
    ```
    
2. Convert each video to frames at 30 fps:
    ```
    ffmpeg -i <path_to_video_from_pevid-uhd>.mp4 -vf fps=30 frame_%d.jpg
    ```

3. Change the variables ```input_gt_file``` and ```output_folder``` paths of your PEViD-UHD annotations 
in [convert_pevid_to_voc.sh]() to match the annotations you want to extract

4. Run the [convert_pevid_to_voc.sh]() to produce Pascal-VOC XML annotations:
    ```
    # From <your_path>/ObjectDetectionTraining
    bash PEViD-UHD/convert_pevid_to_voc.sh
    ```

