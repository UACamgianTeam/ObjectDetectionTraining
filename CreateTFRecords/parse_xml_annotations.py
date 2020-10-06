from lxml import etree
import os


def trim_xml_annotations(instance_annotations_path, class_subset):
    """
    Searches through every XML annotation file (Pascal VOC format) in instance_annotations_path and removes bounding
    boxes whose class label is not in class_subset

    Args:
        instance_annotations_path: The path to the root directory for all instance annotations (Object Detection)
            for a given set (could be train, val, test, etc.)
        class_subset: The list of acceptable class names

    Returns:

    """

    # Iterate recursively over annotation files in the dataset
    for root_name, dir_names, file_names in os.walk(instance_annotations_path, topdown=False):
        print(f'Parsing files from {root_name}')
        for i, file_name in enumerate(file_names):
            if os.path.join(root_name, file_name).endswith('.xml'):
                with open(os.path.join(root_name, file_name),'r+') as annotation_file:
                    # Parse the annotation file
                    tree = etree.parse(annotation_file)

                    # Get the root of the annotation file
                    root = tree.getroot()

                    # Iterate over each 'object' (bounding box) in the file
                    for obj in root.findall('object'):

                        # Get the class name of the bbox
                        obj_name = obj.find('name').text

                        # If that class doesn't exist in our subset...
                        if obj_name not in class_subset:
                            # Remove the bbox
                            root.remove(obj)

                    # Finally, write the parsed xml to a file
                    tree.write(os.path.join(root_name, file_name))


def main():
    class_subset = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    instance_val_annotations_path = '../data/coco_2017/annotations/instance_val_annotation'
    instance_train_annotations_path = '../data/coco_2017/annotations/instance_train_annotation'

    # Trim the validation set, then the train set
    trim_xml_annotations(instance_annotations_path=instance_val_annotations_path,
                         class_subset=class_subset)
    trim_xml_annotations(instance_annotations_path=instance_train_annotations_path,
                         class_subset=class_subset)


if __name__ == "__main__":
    main()
