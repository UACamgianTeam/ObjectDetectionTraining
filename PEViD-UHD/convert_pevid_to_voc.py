""" Conversion between annotation formats """
import os
import fnmatch
from PIL import Image
from absl import flags, app

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

flags.DEFINE_string('input_gt_file', '../data/PEViD-UHD/walking_day_outdoor_3/Walking_day_outdoor_3_4K_trimmed.xgtf',
                    'Path to input groundtruth bboxes file.\n This has the file extension '
                    '".xgtf"')
flags.DEFINE_string('output_folder', '../data/PEViD-UHD/walking_day_outdoor_3/', 'Path to output Pascal-VOC XML annotations')

FLAGS = flags.FLAGS


def format_header(image_name: str, folder_name: str, path_to_input: str, w: int, h: int) -> str:
    path = path_to_input + folder_name + "/" + image_name
    width = str(w)
    height = str(h)

    string = """<annotation>
	<folder>""" + folder_name + """</folder>
	<filename>""" + image_name + """</filename>
	<path>""" + path + """</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>""" + width + """</width>
		<height>""" + height + """</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>"""

    # print(string)
    return string


def format_object(xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    xmin = str(xmin)
    ymin = str(ymin)
    xmax = str(xmax)
    ymax = str(ymax)

    name = "person"

    string = """
    <object>
		<name>""" + name + """</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>""" + xmin + """</xmin>
			<ymin>""" + ymin + """</ymin>
			<xmax>""" + xmax + """</xmax>
			<ymax>""" + ymax + """</ymax>
		</bndbox>
	</object>"""

    # print(string)
    return string


def format_tail():
    string = """
</annotation>"""
    # print(string)
    return string


def main(unused_argv) -> None:
    """
    Converts PEViD-UHD proprietary annotation to Pascal-VOC XML annotations

    Returns:
        None
    """

    flags.mark_flags_as_required(['input_gt_file', 'output_folder'])

    # Load groundtrouth boxes
    gt_lines = list(open(FLAGS.input_gt_file))
    gt_lines = [i.rstrip().split(' ') for i in gt_lines]
    gt_lines = [[float(j) for j in i] for i in gt_lines]
    print(gt_lines)

    gt_per_frames = {}
    for gt_line in gt_lines:
        # [6.0, 1.0, 1012.15, 800.015, 1108.267, 1070.888]
        # frameNumber, personNumber, bodyLeft, bodyTop, bodyRight, bodyBottom
        # frameNumber, _, xmin, ymax, xmax, ymin
        frame_num = int(gt_line[0])
        if frame_num not in gt_per_frames.keys():
            gt_per_frames[frame_num] = []
        # gt_per_frames[frame_num].append( gt_line[2:5] )

        xmin = gt_line[2]
        ymax = gt_line[5]
        xmax = gt_line[4]
        ymin = gt_line[3]
        # print("xmin, ymin, xmax, ymax", xmin, ymin, xmax, ymax)
        gt_per_frames[frame_num].append([xmin, ymin, xmax, ymax])

    # Target Folder
    split_path = FLAGS.output_folder.split('/')
    folder_name = split_path[-2]
    path_to_input = FLAGS.output_folder[0:-(len(folder_name) + 1)]

    print("output_folder", FLAGS.output_folder)
    print("folder_name", folder_name)
    print("path_to_input", path_to_input)

    # for image_name in image_names
    files = sorted(os.listdir(FLAGS.output_folder))
    image_names = fnmatch.filter(files, '*.jpg')
    # print(image_names)

    # WIDTH AND HEIGHT is shared
    first_image = FLAGS.output_folder + image_names[0]
    img = Image.open(first_image)
    w, h = img.size

    # format_header(image_name, folder_name, path_to_input, w, h)
    # format_object(xmin, ymin, xmax, ymax)
    # format_object()
    # format_tail()

    keys = list(gt_per_frames.keys())
    for key in keys:
        print("frame", key, "(", len(gt_per_frames[key]), "): ", gt_per_frames[key])

        objects_parameters = gt_per_frames[key]  # list of [xmin, ymin, xmax, ymax]

        # key "6" to jpg name "0005.jpg" (maybe)
        # key "161" to jpg name "0160.jpg"
        number = key - 1

        name = str(number).zfill(4)
        image_name = name + ".jpg"
        xml_name = name + ".xml"
        print(image_name)
        # print(objects_parameters)

        string = ""
        string += format_header(image_name, folder_name, path_to_input, w, h)

        for object in objects_parameters:
            xmin, ymin, xmax, ymax = object
            string += format_object(xmin, ymin, xmax, ymax)

        string += format_tail()

        # now safe into filename.xml
        # print(string)

        with open(FLAGS.output_folder + xml_name, "w") as text_file:
            text_file.write(string)


if __name__ == "__main__":
    app.run(main)
