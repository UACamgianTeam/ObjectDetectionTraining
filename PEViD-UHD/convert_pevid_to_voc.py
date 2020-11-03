""" Conversion between annotation formats """
import os
import fnmatch
from PIL import Image
from absl import flags, app
import xml.etree.ElementTree as ET
import re
from typing import List
from xml.dom import minidom

"""
# FROM FORMAT
where all annotations for all frames in a video are stored in one .xgtf file
<data>
<sourcefile filename="Exchanging_bags_day_indoor_1_4K.avi">
<file id="0" name="Information">
    <attribute name="SOURCETYPE"/>
    <attribute name="NUMFRAMES"><data:dvalue value="390"/></attribute>
    <attribute name="FRAMERATE"><data:fvalue value="1.0"/></attribute>
    <attribute name="H-FRAME-SIZE"><data:dvalue value="1920"/></attribute>
    <attribute name="V-FRAME-SIZE"><data:dvalue value="1080"/></attribute>
</file>
<object framespan="1:390" id="0" name="Person"> <<< there can be multiple
    <attribute name="box">
        <data:bbox framespan="1:8" height="101" width="39" x="881" y="479"/>
        <data:bbox framespan="9:10" height="101" width="39" x="879" y="479"/>
        <data:bbox framespan="11:19" height="101" width="37" x="879" y="479"/>
        ... etc
"""

"""
# TO FORMAT
where each image x.jpg is accompanied by annotation in x.xml
<annotation>
    <folder>frames_20171126</folder>
    <filename>0001.jpg</filename>
    <path>/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PittsMine/input/frames_20171126/0001.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>3840</width>
        <height>2160</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>

    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>52</xmin>
            <ymin>883</ymin>
            <xmax>172</xmax>
            <ymax>1213</ymax>
        </bndbox>
    </object>
    ... # multiple objects <object> ... </object>
</annotation>
"""

flags.DEFINE_string('input_gt_file', '../data/PEViD-UHD/walking_day_outdoor_3/Walking_day_outdoor_3_4K.xgtf',
                    'Path to input groundtruth bboxes file.\n This has the file extension ".xgtf"')
flags.DEFINE_string('file_prefix', 'frame_',
                    'Perfix for output Pascal-VOC XML annotations')
flags.DEFINE_string('output_folder', '../data/PEViD-UHD/walking_day_outdoor_3/',
                    'Path to output Pascal-VOC XML annotations')
FLAGS = flags.FLAGS

label_map = {
    'Person': 1,
    'Accessory': 2
}


def get_xmlns(xml_file: str) -> str:
    """
    Gets the xmlns for a given XML file.

    Args:
        xml_file: Path to an XML file

    Returns:
        xmlns: The default xmlns (XML namespace) which is prepended to every attribute in the ElementTree
    """
    root = ET.parse(xml_file).getroot()
    xmlns = ''
    m = re.search('{.*}', root.tag)
    if m:
        xmlns = m.group(0)
    return xmlns


def convert_bbox_to_xmls(bbox: ET.Element, class_name: str, class_id: int, file_prefix: int, output_folder: int,
                         height: int, width: int, prettify: bool = True) -> None:
    """
    Converts a given bounding box to an XML file and prettifies the output for human-readability

    Args:
        bbox: An ET.Element object representing a single bounding box element in XML
        class_name: The name of the class for this bounding box
        class_id: The class ID for this bounding box
        file_prefix: The string to prepend to the output XML files
        output_folder: The output folder for saving output XML files
        height: The height in pixels of the input frames
        width: The width in pixels of the input frames
        prettify: True if we wish to pretty-print the saved XML document

    Returns:
        None

    """

    # Get the start and end frames of this bbox
    start_frame, end_frame = tuple(int(item) for item in bbox.attrib['framespan'].split(':'))

    # Get the PEViD-UHD coordinates from this box
    pevid_coordinates = [int(bbox.attrib['height']), int(bbox.attrib['width']), int(bbox.attrib['x']),
                         int(bbox.attrib['y'])]
    # print(f'pevid_coordinates: {pevid_coordinates}')

    # Convert the PEViD-UHD coordinates to Pascal-VOC coordinates
    pascal_coordinates = pevid_coordinates_to_voc(pevid_coordinates)
    # print(f'pascal_coordinates: {pascal_coordinates}')

    # Iterate over every frame for this bbox, convert to Pascal-VOC, and save to files
    for frame_num in range(start_frame, end_frame + 1):
        filename = file_prefix + str(frame_num) + '.xml'
        output_file = os.path.join(output_folder, filename)
        if not os.path.exists(output_file):
            xml = construct_xml_document(pascal_coordinates, class_name, class_id, output_file, height, width)
        else:
            xml = append_to_xml_document(pascal_coordinates, class_name, class_id, output_file)

        # Write the XML annotation to a file
        if prettify:
            pass
            xmlstr = minidom.parseString(ET.tostring(xml)).toprettyxml(indent="   ")
            with open(output_file, "w+") as file:
                file.write(xmlstr.encode('utf-8'))
        else:
            xml.write(output_file)


def get_folder_or_file_name(path: str) -> str:
    """
    Extracts the last folder or file name from a path
    Args:
        path: The entire path to a folder or a file

    Returns:
        folder_name: The last folder or file name from the path
    """
    without_extra_slash = os.path.normpath(path)
    last_part = os.path.basename(without_extra_slash)
    return last_part


def append_to_xml_document(pascal_coordinates: List[int], class_name: str, class_id: int,
                           output_file_path: str) -> ET.ElementTree:
    """
    Appends a new bounding box to an already-existing Pascal-VOC XML annotation file
    Args:
        pascal_coordinates: The Pascal-VOC bounding box
            coordinates (x-top left, y-top left, x-bottom right, y-bottom right)
        class_name: The class name of the annotated object
        class_id: The class ID of the annotated object
        output_file_path: The path + filename of the XML file to be created (the frame number is embedded within the
            output filename)


    Returns:
        xml_tree: The tree representing the XML annotation file
    """
    # Get root of "tree" representing the XML file
    root = ET.parse(output_file_path).getroot()

    # Create new object element as <object>...</object>
    object_element = ET.Element("object")
    root.append(object_element)
    # Create name subelement
    name_element = ET.SubElement(object_element, "name")
    name_element.text = class_name
    # Create bounding box subelement and add coordinates
    bndbox_element = ET.SubElement(object_element, "bndbox")
    xmin_element = ET.SubElement(bndbox_element, "xmin")
    xmin_element.text = str(pascal_coordinates[0])
    ymin_element = ET.SubElement(bndbox_element, "ymin")
    ymin_element.text = str(pascal_coordinates[3])
    xmax_element = ET.SubElement(bndbox_element, "xmax")
    xmax_element.text = str(pascal_coordinates[2])
    ymax_element = ET.SubElement(bndbox_element, "ymax")
    ymax_element.text = str(pascal_coordinates[1])

    xml_tree = ET.ElementTree(root)
    return xml_tree


def construct_xml_document(pascal_coordinates: List[int], class_name: str, class_id: int, output_file_path: str,
                           height: int, width: int) -> ET.ElementTree:
    """
    Creates a new XML annotation file for a particular frame, and adds one bounding box

    Args:
        pascal_coordinates: The Pascal-VOC bounding box
            coordinates (x-top left, y-top left, x-bottom right, y-bottom right)
        class_name: The class name of the annotated object
        class_id: The class ID of the annotated object
        output_file_path: The path + filename of the XML file to be created (the frame number is embedded within the
            output filename)
        height: The height in pixels of each input frame
        width: The width in pixels of each input frame

    Returns:
        xml_tree: The tree representing the XML annotation file
    """
    # Create root element as <annotation>...</annotation>
    root = ET.Element("annotation")

    # Create folder element as <folder>...</folder>
    folder_element = ET.Element("folder")
    folder_element.text = get_folder_or_file_name(FLAGS.output_folder)
    root.append(folder_element)

    # Create filename element as <filename>...</filename>
    filename_element = ET.Element("filename")
    filename_element.text = get_folder_or_file_name(output_file_path)
    root.append(filename_element)

    # Create source element as <source>...</source>
    source_element = ET.Element("source")
    root.append(source_element)
    # Add subelements to <source> element
    database_element = ET.SubElement(source_element, "database")
    database_element.text = "PEViD-UHD"
    annotation_element = ET.SubElement(source_element, "annotation")
    annotation_element.text = "PEViD-UHD"
    image_element = ET.SubElement(source_element, "image")
    image_element.text = "EPFL"
    url_element = ET.SubElement(source_element, "url")
    url_element.text = "https://alabama.app.box.com/folder/124516925859"

    # Create size element as <size>...</size>
    size_element = ET.Element("size")
    root.append(size_element)
    # Add subelemnts to <size> element
    width_element = ET.SubElement(size_element, "width")
    width_element.text = str(width)
    height_element = ET.SubElement(size_element, "height")
    height_element.text = str(height)
    depth_element = ET.SubElement(size_element, "depth")
    depth_element.text = str(3)

    # Create segmented element as <segmented>...</segmented>
    segmented_element = ET.Element("segmented")
    segmented_element.text = str(0)
    root.append(segmented_element)

    # Create object element as <object>...</object>
    object_element = ET.Element("object")
    root.append(object_element)
    # Create name subelement
    name_element = ET.SubElement(object_element, "name")
    name_element.text = class_name
    # Create bounding box subelement and add coordinates
    bndbox_element = ET.SubElement(object_element, "bndbox")
    xmin_element = ET.SubElement(bndbox_element, "xmin")
    xmin_element.text = str(pascal_coordinates[0])
    ymin_element = ET.SubElement(bndbox_element, "ymin")
    ymin_element.text = str(pascal_coordinates[3])
    xmax_element = ET.SubElement(bndbox_element, "xmax")
    xmax_element.text = str(pascal_coordinates[2])
    ymax_element = ET.SubElement(bndbox_element, "ymax")
    ymax_element.text = str(pascal_coordinates[1])

    xml_tree = ET.ElementTree(root)
    return xml_tree


def pevid_coordinates_to_voc(pevid_coordinates: List[int]) -> List[int]:
    """
    Converts PEViD-UHD coordinates (height, width, x, y) to
    Pascal-VOC coordinates (x-top left, y-top left, x-bottom right, y-bottom right)
    Args:
        pevid_coordinates: Bounding box coordinates for an object in the PEViD-UHD style

    Returns:
        voc_coordinates: The Pascal-VOC style coordinates for the object
    """
    x_top_left = pevid_coordinates[2]
    y_top_left = pevid_coordinates[3] + pevid_coordinates[0]
    x_bottom_right = pevid_coordinates[2] + pevid_coordinates[1]
    y_bottom_right = pevid_coordinates[3]
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]


def main(unused_argv) -> None:
    """
    Converts PEViD-UHD proprietary annotation to Pascal-VOC XML annotations

    Returns:
        None
    """
    # Get root of "tree" representing the XML file
    root = ET.parse(FLAGS.input_gt_file).getroot()

    # Get the namespace prefix of the XML file
    ns = get_xmlns(FLAGS.input_gt_file)

    # Find height and width in pixels
    height = root.find(f"./{ns}data/{ns}sourcefile/{ns}file/*[@name='V-FRAME-SIZE']").find('*').attrib['value']
    width = root.find(f"./{ns}data/{ns}sourcefile/{ns}file/*[@name='H-FRAME-SIZE']").find('*').attrib['value']

    # Find each object
    for annotation_element in root.findall(f'./{ns}data/{ns}sourcefile/{ns}object'):

        # Get class name and associated class id from annotation
        class_name = annotation_element.attrib['name']
        class_id = label_map[class_name]
        # print(f'class_id = {class_id}, class_name = {class_name}')

        # Find each bounding box across frames and convert to Pascal-VOC XML
        for bbox in annotation_element.find(f"./*[@name='box']"):
            convert_bbox_to_xmls(bbox, class_name, class_id, FLAGS.file_prefix, FLAGS.output_folder, height, width)


if __name__ == "__main__":
    app.run(main)
