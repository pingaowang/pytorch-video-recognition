import os
import numpy as np
from PIL import Image


def show_a_input(tensor_inputs):
    data_norm_para = [127.5, 127.5, 127.5]
    for i in range(4):
        arr = tensor_inputs[i,:,0,:,:].data.numpy() + np.array(data_norm_para)[:, np.newaxis, np.newaxis]
        arr = arr.astype(np.uint8).transpose((1, 2, 0))
        Image.fromarray(arr).show()

