
import os
from PIL import Image
import glob
import cv2

WSI_MASK_PATH = "/media/lab411/机械盘1/Dataset/KITTI分割数据集/data_semantics/training/semantic_rgb/"

paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))

for path in paths:
    img= cv2.imread(path)
    img.show()

