import re
import cv2


def get_filename_suffix(path):
    """
    1. file's name; 2. suffix;
    :param path: video's path, e.g. 'xxx/yyy/zzz.avi'
    :return:
        file_name : str = 'zzz'
        suffix : str = 'avi'
    """
    suffix = re.split('\.', path)[-1]
    video_filename = re.split('\/', re.split('\.', path)[0])[-1]
    return video_filename, suffix


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



