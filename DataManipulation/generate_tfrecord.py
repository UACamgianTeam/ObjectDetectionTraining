""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv_from_pevid(path):
    """Iterates through all .xml files in a given directory and combines
    them in a single Pandas dataframe.
    This is used for the PEViD-UHD dataset, which has a rather odd directory structure, where each volume is
    stored as a separate subdirectory.

    NOTE: The objects stored in the XMLs created from the JSON annotations from COCO are missing fields like
    "pose", "truncated", and "difficult", so indexing the member object from "root.findall('object')" will act
    differently in this function based upon how many fields are specified in each <object> tag in the XML annotation
    files.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    counter = 0
    xml_list = []

    # Loop over all xml annotation files
    for xml_file in glob.iglob(f'{path}/**/annotations/*.xml', recursive=True):
        # TODO: DELETE THIS (this is for creating TRAIN and VAL TFRecords##########
        if 'exchanging_bags_day_indoor_1' not in xml_file:
            continue
        ###########################################################################
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     # class name
                     member[0].text,
                     # bbox coordinates
                     float(member[1][0].text),  # x
                     float(member[1][1].text),  # y
                     float(member[1][2].text),  # width
                     float(member[1][3].text)  # height
                     )
            xml_list.append(value)
        counter += 1

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml_to_csv_with_class_subdirs(path):
    """Iterates through all .xml files in a given directory and combines
    them in a single Pandas dataframe.
    This can be used when the .xml files are separated into different subdirectories, each subdirectory being
    a class name, like the following: "annotations/instance_val_annotation/airplane/*.xml"

    NOTE: The objects stored in the XMLs created from the JSON annotations from COCO are missing fields like
    "pose", "truncated", and "difficult", so indexing the member object from "root.findall('object')" will act
    differently in this function based upon how many fields are specified in each <object> tag in the XML annotation
    files.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    counter = 0
    xml_list = []

    for xml_file in glob.iglob(f'{path}/**/*.xml', recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     # class name
                     member[0].text,
                     # bbox coordinates
                     float(member[1][0].text),  # x
                     float(member[1][1].text),  # y
                     float(member[1][2].text),  # width
                     float(member[1][3].text)  # height
                     )
            xml_list.append(value)
        counter += 1

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml_to_csv_from_labelImg(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    counter = 0
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        if not counter % 100:
            print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
        counter += 1
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, is_pevid: bool = False):
    """Creates a TF example given a dataframe containing annotation information and input image path.
     This is only used for TFExample creation from the PEViD-UHD dataset"""

    if is_pevid:
        # Get and add the volume and 'frames' to the path
        volume_name = group.filename.split('_frame')[0]
        path = os.path.join(path, volume_name, 'frames')

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tfrecord(output_path: str, image_dir: str, xml_dir: str, csv_path: str = None,
                      is_pevid: bool = False) -> None:
    """
    Generates a TFRecord from Pascal VOC XML annotations

    Args:
        output_path: The path to output the TFRecord file
        image_dir: The path to the folder where the input image files are stored.
            Defaults to the same as xml_dir
        xml_dir: The path to the folder where input .xml files are located
        csv_path: The path to the output .csv file. If None is provided, no file
            will be written
        is_pevid: True if the dataset to generate a TFRecord file from is the PEViD-UHD
            dataset, false otherwise.

    Returns:
        None
    """
    # Create a TFRecordWriter object (used to create TFRecords)
    writer = tf.python_io.TFRecordWriter(output_path)

    # Iterate through all XML files and create single Pandas dataframe containing all image annotations
    if is_pevid:
        examples = xml_to_csv_from_pevid(xml_dir)
    else:
        examples = xml_to_csv_with_class_subdirs(xml_dir)

    # Split up the examples by filename
    grouped = split(examples, 'filename')

    # Iterate through each file and create TF examples for each
    print(f'Creating TFRecord file: {output_path}')
    for group in grouped:
        # Create a TF example object containing image AND annotation
        tf_example = create_tf_example(group, image_dir, is_pevid)
        # Write the TF example object into the TFRecordWriter object
        writer.write(tf_example.SerializeToString())

    # Close the TFRecordWriter object
    writer.close()
    print(f'Successully created the TFRecord file: {output_path}')

    # Save the CSV file (if we wish to do so)
    if csv_path is not None:
        examples.to_csv(csv_path, index=None)
        print(f'Successfully created the CSV file: {csv_path}')


if __name__ == '__main__':
    generate_tfrecord(args.output_path, args.image_dir, args.xml_dir, args.csv_path, is_pevid=True)
