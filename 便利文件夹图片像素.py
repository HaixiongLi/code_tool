
import os
from PIL import Image
import glob
import cv2

WSI_MASK_PATH = "/media/lab411/机械盘1/Dataset/KITTI分割数据集/data_semantics/training/semantic_rgb/"

pre_path = '/media/lab411/机械盘1/Dataset/KITTI分割数据集/data_semantics/training/semantic_grey/'

paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))


def access_pixels(name,frame):
    print(frame.shape)  # shape内包含三个元素：按顺序为高、宽、通道数
    height = frame.shape[0]
    weight = frame.shape[1]
    channels = frame.shape[2]
    print("weight : %s, height : %s, channel : %s" % (weight, height, channels))

    for row in range(height):  # 遍历高
        for col in range(weight):  # 遍历宽
            if frame[row, col, 0] == 142:  # 全部像素取反，实现一个反向效果
                frame[row, col, :] = 255
                # frame[row, col, 1] == 255  # 全部像素取反，实现一个反向效果
                # frame[row, col, 2] == 255  # 全部像素取反，实现一个反向效果
            else:
                frame[row, col, :] = 0
    savename = pre_path+path[-13:]
    cv2.imwrite(savename,frame)
    print("saved"+ savename)
    # cv2.imshow("fanxiang", frame)



    # for row in range(height):  # 遍历高
    #     for col in range(weight):  # 遍历宽
    #         for c in range(channels):  # 便利通道
    #             if c == 0:
    #                 if frame[row, col, c] == 142:  # 全部像素取反，实现一个反向效果
    #                     if frame[row, col, c] == 255  # 全部像素取反，实现一个反向效果
    #             else:
    #                 frame[row, col, c] = 0
    # cv2.imshow("fanxiang", frame)

for path in paths:
    img= cv2.imread(path)
    cv2.imshow("Picture", img)
    access_pixels(path,img)

cv2.waitKey(0)




