import xml.etree.ElementTree as ET
import glob
import os


def fix_pevid_filenames(path):
    """
    throwaway function, just fixing up PEVID filenames

    Args:
        path: str
            The path containing the .xml files

    Returns:
        None
    """
    # Loop over all .jpgs, fixing their filename
    for jpg_file in glob.iglob(f'{path}/**/frames/*.jpg', recursive=True):
        # Split up the path into its components
        path_components = jpg_file.split(os.sep)
        # Get the volume name from the full path
        volume_name = path_components[-3]
        # Set the new filename
        if volume_name not in path_components[-1]:
            new_jpg_file = os.path.join(*path_components[:-1], f'{volume_name}_{path_components[-1]}')
            # Rename the file
            os.rename(src=jpg_file, dst=new_jpg_file)

    # Loop over all .xmls, fixing their filename
    for xml_file in glob.iglob(f'{path}/**/annotations/*.xml', recursive=True):
        # Split up the path into its components
        path_components = xml_file.split(os.sep)
        # Get the volume name from the full path
        volume_name = path_components[-3]
        # Set the new filename
        if volume_name not in path_components[-1]:
            new_xml_file = os.path.join(*path_components[:-1], f'{volume_name}_{path_components[-1]}')
            # Rename the file
            os.rename(src=xml_file, dst=new_xml_file)

    # Finally. fix the filename elements in each annotation file
    fix_pevid_xml_filename_elements(path)


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
        # Get the file name from the full path
        _, filename = os.path.split(xml_file)
        filename, _ = os.path.splitext(filename)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root.find('filename').text = f'{filename}.jpg'
        tree.write(xml_file)


if __name__ == "__main__":
    pevid_path = "../../data/PEViD-UHD"
    # fix_pevid_filenames(pevid_path)
    fix_pevid_xml_filename_elements(pevid_path)

