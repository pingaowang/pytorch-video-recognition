import os
import cv2
from utils import get_prop, get_filename_suffix
"""
The input root contains class folders, which have all videos of this class.
Process all videos to a new jpg root: root/class/video_name/{i}.jpg
"""
input_video_dir = '../data/KTH_video'
output_dir = '../data/KTH_jpg_0'
assert not os.path.isdir(output_dir), "The output dir exists."
os.mkdir(output_dir)

list_classes = os.listdir(input_video_dir)
if '.DS_Store' in list_classes:
    list_classes.remove('.DS_Store')

for cls in list_classes:
    path_cls = os.path.join(input_video_dir, cls)
    output_path_cls = os.path.join(output_dir, cls)
    os.mkdir(output_path_cls)
    list_videos = os.listdir(path_cls)
    for video_path in list_videos:
        capture = cv2.VideoCapture(os.path.join(path_cls, video_path))
        v_name, v_suffix = get_filename_suffix(video_path)
        v_count, v_fps, v_weight, v_hight = get_prop(capture)

        output_jpg_folder = os.path.join(output_path_cls, v_name)
        os.mkdir(output_jpg_folder)

        count = 0
        i = 0
        retaining = True
        print(v_name)
        while (count < v_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            cv2.imwrite(filename=os.path.join(output_jpg_folder, '{}.jpg'.format(i)), img=frame)
            i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()
