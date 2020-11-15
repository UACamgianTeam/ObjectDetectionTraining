# DOTA to COCO

This is a simple converter for the [dota](https://captain-whu.github.io/DOTA/dataset.html) dataset. *dota_utils.py* and *DOTA2COCO.py* are taken from here: https://github.com/CAPTAIN-WHU/DOTA_devkit. *DOTA2COCO.py* is slightly modified to take command-line arguments but that is all.

The DOTA dataset labels their images with .txt labelTxt files in the format:

```
imagesource:GoogleEarth
gsd:0.146343590398
2753 2408 2861 2385 2888 2468 2805 2502 plane 0
3445 3391 3484 3409 3478 3422 3437 3402 large-vehicle 0
3185 4158 3195 4161 3175 4204 3164 4199 large-vehicle 0
2870 4250 2916 4268 2912 4283 2866 4263 large-vehicle 0
630 1674 628 1666 640 1654 644 1666 small-vehicle 0
636 1713 633 1706 646 1698 650 1706 small-vehicle 0
...
190 2701 199 2698 203 2723 194 2724 large-vehicle 0
225 2610 231 2609 234 2626 227 2628 small-vehicle 0
220 2654 228 2652 230 2670 222 2672 small-vehicle 1
1434 1943 1442 1941 1445 1958 1436 1959 large-vehicle 0
1153 1771 1161 1766 1170 1779 1162 1783 small-vehicle 1
1497 2068 1507 2070 1503 2086 1495 2083 small-vehicle 1
631 4431 678 4322 813 4368 755 4485 plane 0
```
where the file structure is:
```
+train
    +images
        -P0006.png
        -P0007.png
        ...
        -P1396.png
    +labelTxt
        -P0006.txt
        -P0007.txt
        ...
        -P1396.txt
```

**DOTA2COCO.py** takes in a training/validation set of images and labelTxts and converts the labels into a corresponding coco json file. Once we have the coco json file, we can treat the dataset as we would a coco dataset because it will be in that format. We can also delete the labelTxts.

## How To run

1. Run `pipenv shell` then `pipenv install` to install all dependencies.
2. Run `python DOTA2COCO.py ../path/to/dataset ../path/to/new/coco.json`
    * Example: `python DOTA2COCO.py ../dota_data/train ../dota_data/annotations/train.json`