import annotation_conversion
import re
import argparse
import json
import os


def main(args):
    """
    Converts COCO JSON annotations to Pascal VOC annotations

    :param args: Command-line arguments
    :return:
    """
    # If the output directory of the annotations doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open the JSON file containing the annotations and get the contents
    content = json.load(open(args.anno_file, 'r'))

    # If the annotation file is an 'instance' file
    if args.type == 'instance':

        # make subdirectories
        sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(args.output_dir, str(sub_dir))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # Parse the JSON into separate XML files for each image
        annotation_conversion.parse_instance(content, args.output_dir, args.database, args.image_source_name, args.source_url)
    elif args.type == 'keypoint':
        annotation_conversion.parse_keypoints(content, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument("--type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'],
                        default='instance')
    parser.add_argument("--database", type=str, help="The name of the image database",
                        choices=['PEViD-UHD', 'DOTA', 'COCO2017'], default='PEViD-UHD')
    parser.add_argument("--source_url", type=str, help="source URL of images",
                        default='https://alabama.app.box.com/folder/125463320422')
    parser.add_argument("--image_source_name", type=str, help="Name of image source",
                        default='EPFL')
    parser.add_argument("--output_dir", help="output directory for voc annotation xml file")
    args = parser.parse_args()
    main(args)