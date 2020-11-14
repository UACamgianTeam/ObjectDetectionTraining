"""
Contains scripts for converting between common annotation formats
- TensorFlow TFRecord
- Pascal VOC XML
- COCO JSON
"""

import argparse
import json
import cytoolz
from lxml import etree, objectify
import os
import re


def instance2xml_base(annotation, folder_name: str, db_name: str, image_source_name: str,
                      source_url: str) -> etree.ElementTree(any):
    """
    Converts the base of an image annotation into an Pascal VOC XML Tree.
    This base contains all of the metadata about the annotation, and
    doesn't include all of the bounding boxes

    Args:
        annotation: A base annotation for a single image, containing metadata
        folder_name: The name of the paprent folder of the image
        db_name: The name of the image database
        image_source_name: The name of the source for the image
        source_url: The source URL at which the image(s) can be found

    Returns:
        A filled XML tree representing an XML instance annotation file
    """
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        # E.folder('VOC2014_instance/{}'.format(annotation['category_id'])),
        E.folder(f"{folder_name}_instance/{annotation['category_id']}"),
        E.filename(annotation['file_name']),
        E.source(
            E.database(db_name), # was 'MS COCO 2014'
            E.annotation(db_name), # was 'MS COCO 2014'
            E.image(image_source_name), # was 'Flickr'
            E.url(source_url)
        ),
        E.size(
            E.width(annotation['width']),
            E.height(annotation['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content: json, outdir: str) -> None:
    """
    Parses all annotations instances and saves them into separate XML annotation files
    for each image.
    Args:
        content: JSON content containing all annotation information
        outdir: The directory under which to store XML annotation files

    Returns:
        None
    """
    categories = {d['id']: d['name'] for d in content['categories']}
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations'])))
    # convert category id to name
    for instance in merged_info_list:
        instance['category_id'] = categories[instance['category_id']]
    # group by filename to pool all bbox in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        anno_tree = instance2xml_base(groups[0], outdir, args.database, args.image_source_name, args.source_url)
        # if one file have multiple different objects, save it in each category sub-directory
        filenames = []
        for group in groups:
            filenames.append(os.path.join(outdir, re.sub(" ", "_", group['category_id']),
                                    os.path.splitext(name)[0] + ".xml"))
            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)
        print("Formating instance xml file {} done!".format(name))


def keypoints2xml_base(anno):
    """
    Converts the base of a keypoint annotation into Pascal VOC XML format,
    and returns that formatted XML tree

    Args:
        anno: The keypoint annotation containing metadata about the file

    Returns:
        A filled XML tree representing an XML keypoint annotation file
    """
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "VOC2014_keypoints"
    etree.SubElement(annotation, "filename").text = anno['file_name']
    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "MS COCO 2014"
    etree.SubElement(source, "annotation").text = "MS COCO 2014"
    etree.SubElement(source, "image").text = "Flickr"
    etree.SubElement(source, "url").text = anno['coco_url']
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(anno["width"])
    etree.SubElement(size, "height").text = str(anno["height"])
    etree.SubElement(size, "depth").text = '3'
    etree.SubElement(annotation, "segmented").text = '0'
    return annotation


def keypoints2xml_object(anno, xmltree, keypoints_dict, bbox_type='xyxy'):
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    key_object = etree.SubElement(xmltree, "object")
    etree.SubElement(key_object, "name").text = anno['category_id']
    bndbox = etree.SubElement(key_object, "bndbox")
    etree.SubElement(bndbox, "xmin").text = str(xmin)
    etree.SubElement(bndbox, "ymin").text = str(ymin)
    etree.SubElement(bndbox, "xmax").text = str(xmax)
    etree.SubElement(bndbox, "ymax").text = str(ymax)
    etree.SubElement(key_object, "difficult").text = '0'
    keypoints = etree.SubElement(key_object, "keypoints")
    for i in range(0, len(keypoints_dict)):
        keypoint = etree.SubElement(keypoints, keypoints_dict[i+1])
        etree.SubElement(keypoint, "x").text = str(anno['keypoints'][i*3])
        etree.SubElement(keypoint, "y").text = str(anno['keypoints'][i*3+1])
        etree.SubElement(keypoint, "v").text = str(anno['keypoints'][i*3+2])
    return xmltree


def parse_keypoints(content, outdir):
    keypoints = dict(zip(range(1, len(content['categories'][0]['keypoints'])+1), content['categories'][0]['keypoints']))
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations']))
    # convert category name to person
    for keypoint in merged_info_list:
        keypoint['category_id'] = "person"
    # group by filename to pool all bbox and keypoint in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        filename = os.path.join(outdir, os.path.splitext(name)[0]+".xml")
        anno_tree = keypoints2xml_base(groups[0])
        for group in groups:
            anno_tree = keypoints2xml_object(group, anno_tree, keypoints, bbox_type="xyxy")
        doc = etree.ElementTree(anno_tree)
        doc.write(open(filename, "w"), pretty_print=True)
        print("Formating keypoints xml file {} done!".format(name))


def main(args):

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
        parse_instance(content, args.output_dir)
    elif args.type == 'keypoint':
        parse_keypoints(content, args.output_dir)


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