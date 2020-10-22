
# This is the first step for kitti segmentation
# In this script, we resize all images to (1242,375) and generate the class label for all the pixels
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os
from collections import namedtuple
from collections import namedtuple

# save_dir = '/Users/lihaixiong/Documents/Python/semantic_tmp/FCN_Kitti-master/Dataset/semantic_image/training/id/'
save_dir = '/home/lab2/work/lhx/data/data_semantics/training/id/'
Img_number_dir=os.listdir(save_dir)

for i in range(len(Img_number_dir)):

    print (i)
    print(Img_number_dir[i])
    a = Image.open(save_dir+Img_number_dir[i])
    # cv2.imshow("show",a)
    # cv2.waitKey()
    a = np.array(a)
    print(a.min(),a.max())
    if(a.max()>19):
        print('error')
    # for m in range(np.shape(a)[0]):
    #     for n in range(np.shape(a)[1]):
    #         if(a[m,n]>19):
    #             print(m,n)



    # np.save(save_dir+Img_number_dir[i], label_img)

