import os
from shutil import copytree
from random import randint

validation_persentage = 15

root_input = "/home/pingao/datasets/ucf101_as_frames"
root_output = "data/ucf101_v2"
l_folder_level_1 = ['train', 'val', 'test']
l_cls = os.listdir(os.path.join(root_input, 'train'))

# build output folder structure
os.makedirs(root_output, exist_ok=True)
for folder_level_1 in l_folder_level_1:
    os.makedirs(os.path.join(root_output, folder_level_1), exist_ok=True)
    for cls in l_cls:
        os.makedirs(os.path.join(root_output, folder_level_1, cls), exist_ok=True)

l_t_subfolder_cls_group = list()
for folder_level_1 in l_folder_level_1:
    for cls in l_cls:
        path_folder = os.path.join(root_input, folder_level_1, cls)
        l_subfolders = os.listdir(path_folder)
        l_path_subfolders = [os.path.join(path_folder, x) for x in l_subfolders]
        for path_subfolders in l_path_subfolders:
            group = int(os.path.basename(path_subfolders).split('_')[2][1:])
            l_t_subfolder_cls_group.append(
                (path_subfolders, cls, group)
            )

# copy
for minimum_folder in l_t_subfolder_cls_group:
    path, cls, group = minimum_folder
    # to train or val
    if group <= 20:
        # to val
        if randint(0, 100) <= validation_persentage:
            folder_level_1 = "val"
        else:
            folder_level_1 = "train"
    # to test
    else:
        folder_level_1 = "test"

    path_copy_dir = os.path.join(root_output, folder_level_1, cls, os.path.basename(path))
    if not os.path.isdir(path_copy_dir):
        copytree(path, path_copy_dir)







