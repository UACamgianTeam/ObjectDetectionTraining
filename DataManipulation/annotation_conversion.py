"""
Contains scripts for converting between common annotation formats
- TensorFlow TFRecord
- Pascal VOC XML
- COCO JSON
"""

import cytoolz
from lxml import etree, objectify
import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import glob
from coco_assistant import COCO_Assistant


def merge_coco_datasets(img_dir: str, ann_dir: str) -> None:
    """
    Merges COCO-style annotations into a single COCO JSON annotation file

    Parameters:
    -----------
    img_dir: Path to directory containing image folders
    ann_dir: Path to directory containing COCO-JSON annotations

    Returns:
    --------
    None
    """
    # Create a COCO Assistant object from the coco-assistant package and merge
    # the datasets
    coco_assistant = COCO_Assistant(img_dir, ann_dir)
    coco_assistant.merge(merge_images=True)


def convert_coco_to_pascal(anno_file: str,
                           anno_type: str = 'instance',
                           database_name: str = 'PEViD-UHD',
                           source_url: str = 'https://alabama.app.box.com/folder/125463320422',
                           image_source_name: str = 'EPFL',
                           output_dir: str = None):
    """
    Converts COCO JSON annotations to Pascal VOC XML annotations

    Params:
    --------
        anno_file: annotation file for object instance/keypoint
        anno_type: object instance or keypoint (what type of annotations are these?)
        database_name: The name of the image database
        source_url: The source URL of the images
        image_source_name: The name of the image source
        output_dir: output directory for voc annotation xml file

    Returns:
    ---------
        None
    """
    # If the output directory of the annotations doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the JSON file containing the annotations and get the contents
    content = json.load(open(anno_file, 'r'))

    # If the annotation file is an 'instance' file
    if anno_type == 'instance':

        # make subdirectories
        sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(output_dir, str(sub_dir))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # Parse the JSON into separate XML files for each image
        parse_instance(content, output_dir, database_name, image_source_name, source_url)
    elif anno_type == 'keypoint':
        parse_keypoints(content, output_dir)


def convert_pascal_to_coco(labels_path: str = None,
                           annotation_dir_path: str = None,
                           annotation_ids_file_path: str = None,
                           extension: str = '',
                           annotation_paths_list_path: str = None,
                           output_file_path: str = 'output.json'):
    """
    Converts Pascal VOC XML annotations to COCO JSON annotations

    Params:
    -------
        labels_path: Path to label list
        annotation_dir_path: Path to annotation files directory
            This is not needed when annotation_paths_list_path is passed (not None)
        annotation_ids_file_path: Path to annotation files IDs list
            This is not needed when annotation_paths_list_path is passed (not None)
        extension: Additional extension of annotation file
        annotation_paths_list_path: Path to annotation paths list.
            This is not needed when annotation_dir_path and annotation_ids_file_path are passed (not None)
        output_file_path: Path to output JSON file

    Returns:
    --------
        None
    """
    # Get mapping from class names to class ids
    label2id = get_label2id(labels_path)

    # Get annotation paths
    ann_paths = get_annpaths(
        ann_dir_path=annotation_dir_path,
        ann_ids_path=annotation_ids_file_path,
        ext=extension,
        annpaths_list_path=annotation_paths_list_path
    )

    # Perform conversion
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=output_file_path,
        extract_num_from_imgid=True
    )


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def fix_pevid_xml_filename_elements(path):
    """
    Throwaway function, just changing each filename element of each XML annotation file to be correct

    Args:
        path: str
            The path containing the .xml files

    Returns:
        None
    """

    # Loop over all xml annotation files
    for xml_file in glob.iglob(f'{path}/**/annotations/*.xml', recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        old_filename = root.find('filename').text
        filename, ext = os.path.splitext(old_filename)
        root.find('filename').text = f'{filename}.jpg'
        tree.write(xml_file)


def instance2xml_base(annotation, folder_name: str, db_name: str, image_source_name: str,
                      source_url: str):
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


def instance2xml_bbox(bbox_annotation, bbox_type='xyxy'):
    """
    Converts a bounding box annotation into an XML
    representation of that bounding box

    Bounding Box Format:
        xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)

    Args:
        bbox_annotation: The data for the bounding box annotation
    Returns:
        A filled XML tree representing the bounding box
    """
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = bbox_annotation['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = bbox_annotation['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(bbox_annotation['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(bbox_annotation['iscrowd'])
    )
    return anno_tree


def parse_instance(content: json, outdir: str, db_name: str, image_source_name: str, source_url: str) -> None:
    """
    Parses all annotations instances and saves them into separate XML annotation files
    for each image.
    Args:
        content: JSON content containing all annotation information
        outdir: The directory under which to store XML annotation files
        db_name: The name of the image database
        image_source_name: The name of the source for the image
        source_url: The source URL at which the image(s) can be found

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
        anno_tree = instance2xml_base(groups[0], outdir, db_name, image_source_name, source_url)
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
        print("Formatting keypoints xml file {} done!".format(name))


if __name__ == "__main__":

    coco_dir = '/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/change_detected_PEViD-UHD/separated'

    images_dir = os.path.join(coco_dir, 'images')
    annotations_dir = os.path.join(coco_dir, 'annotations')

    merge_coco_datasets(images_dir, annotations_dir)
