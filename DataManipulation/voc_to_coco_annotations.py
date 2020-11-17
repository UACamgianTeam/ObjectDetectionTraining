import annotation_conversion
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default='../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/annpaths_list.txt',
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default='../data/PEViD-UHD/labels.txt',
                        help='path to label list.')
    parser.add_argument('--output_file', type=str, default='../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    args = parser.parse_args()

    # Convert Pascal VOC annotation to COCO JSON annotations
    annotation_conversion.convert_pascal_to_coco(labels_path=args.labels,
                                                 annotation_dir_path=args.ann_dir,
                                                 annotation_ids_file_path=args.ann_ids,
                                                 extension=args.ext,
                                                 annotation_paths_list_path=args.ann_paths_list,
                                                 output_file_path=args.output_file)
