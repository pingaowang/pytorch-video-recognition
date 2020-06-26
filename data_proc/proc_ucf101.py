import os
from random import randint
import cv2


def get_prop(capture):
    """
    get props of a video
    :param video_path: capture object of video
    :return fpg: float
    """
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, frame_fps, frame_width, frame_height


root_input_video = "/home/pingao/datasets/UCF101/UCF-101"
root_output = "data/ucf101"

l_cls = os.listdir(root_input_video)
l_tvt = ["train", "val", "test"]
persent_val_set = 15

os.makedirs(root_output, exist_ok=True)
os.makedirs(os.path.join(root_output, "train"), exist_ok=True)
os.makedirs(os.path.join(root_output, "val"), exist_ok=True)
os.makedirs(os.path.join(root_output, "test"), exist_ok=True)

for tvt in l_tvt:
    for cls in l_cls:
        os.makedirs(os.path.join(root_output, tvt, cls), exist_ok=True)


for cls in l_cls:
    input_folder = os.path.join(root_input_video, cls)
    l_video_names = os.listdir(input_folder)
    for video_name in l_video_names:
        group = int(video_name.split('_')[2][1:])
        if group <= 20:
            if (randint(0, 100) <= persent_val_set):
                # to val
                tvt = "val"
            else:
                # to train
                tvt = "train"
        else:
            # to test
            tvt = "test"

        path_video = os.path.join(input_folder, video_name)
        path_output_frames_folder = os.path.join(root_output, tvt, cls, video_name)
        if not os.path.isdir(path_output_frames_folder):
            os.makedirs(path_output_frames_folder, exist_ok=True)

            # extract frames
            capture = cv2.VideoCapture(path_video)
            # v_name, v_suffix = get_filename_suffix(video_path)
            v_count, v_fps, v_weight, v_hight = get_prop(capture)

            output_jpg_folder = path_output_frames_folder
            # os.mkdir(output_jpg_folder)

            count = 0
            i = 0
            retaining = True
            print(path_video)
            while (count < v_count and retaining):
                retaining, frame = capture.read()
                if frame is None:
                    continue

                cv2.imwrite(filename=os.path.join(output_jpg_folder, '{}.jpg'.format(str(i).zfill(5))), img=frame)
                i += 1
                count += 1

            # Release the VideoCapture once it is no longer needed
            capture.release()


