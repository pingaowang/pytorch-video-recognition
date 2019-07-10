from exp_config_reader import *

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Users/pingaowang/Google Drive/study/video_classification_research/datasets/UCF-101'
            # Save preprocess data into output_dir
            output_dir = DATA_PATH# '/Users/pingaowang/Google Drive/study/video_classification_research/pytorch-video-recognition/data/ucf101_motion_mini'
            return root_dir, output_dir

        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'
            output_dir = '/path/to/VAR/hmdb51'
            return root_dir, output_dir

        elif database == 'ucf_motion':
            # folder that contains class labels
            root_dir = '/Users/pingaowang/Google Drive/study/video_classification_research/datasets/UCF-101'
            # Save preprocess data into output_dir
            output_dir = DATA_PATH  # '/Users/pingaowang/Google Drive/study/video_classification_research/pytorch-video-recognition/data/ucf101_motion_mini'
            return root_dir, output_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'