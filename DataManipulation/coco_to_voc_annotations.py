import annotation_conversion
import argparse

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
    annotation_conversion.convert_coco_to_pascal(anno_file=args.anno_file,
                                                 anno_type=args.type,
                                                 database_name=args.database,
                                                 source_url=args.source_url,
                                                 image_source_name=args.image_source_name,
                                                 output_dir=args.output_dir)
