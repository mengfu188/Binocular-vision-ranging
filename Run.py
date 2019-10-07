# /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import time
from math import *
from GetPoint import *
from Calibration import *

if __name__ == "__main__":
    start = time.time()
    # 加载分别从两个摄像头传入的图像
    IMG_LETF = cv2.imread('./assets/IMG_LEFT.jpg')
    IMG_RIGHT = cv2.imread('./assets/IMG_RIGHT.jpg')
    # 获取图像矩阵
    # ROWS, COLS, CHANNELS = IMG_LETF.shape
    # 颜色空间转换
    IMG_LETF = cv2.cvtColor(IMG_LETF, cv2.COLOR_BGR2RGB)
    IMG_RIGHT = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_BGR2RGB)
    # 转换为灰度图
    IMG_LEFT_GRAY = cv2.cvtColor(IMG_LETF, cv2.COLOR_RGB2GRAY)
    IMG_RIGHT_GRAY = cv2.cvtColor(IMG_RIGHT, cv2.COLOR_RGB2GRAY)
    end = time.time()

    # 连线匹配的点
    # MatchPoint(IMG_LETF,IMG_RIGHT,GetPoint(IMG_LEFT_GRAY,IMG_RIGHT_GRAY)[2])
    (IMG_GRAY, obj_points, img_points, RET, MTX, DIST, RVS, TVS) = Calibration("./assets/calibration/*.JPG", 7, 7, 1024,
                                                                               768)
    # 内参数矩阵
    K = MTX
    # 基础矩阵
    F = GetPoint(IMG_LEFT_GRAY, IMG_RIGHT_GRAY)[3]
    # 本征矩阵
    E = K.T * F * K

    print('totally costs: %f' % (end - start))
    # 连线匹配的点
    # MatchPoint(IMG_LETF, IMG_RIGHT, GetPoint(IMG_LEFT_GRAY, IMG_RIGHT_GRAY))

    # 以下参考https://blog.csdn.net/qq_39938666/article/details/91126944
    # 两个trackbar用来调节不同的参数查看效果
    # num = cv2.getTrackbarPos("num", "depth")
    # blockSize = cv2.getTrackbarPos("blockSize", "depth")
    # if blockSize % 2 == 0:
    #     blockSize += 1
    # if blockSize < 5:
    #     blockSize = 5

    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create()
    disparity = stereo.compute(IMG_LEFT_GRAY, IMG_RIGHT_GRAY)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("left", IMG_LETF)
    cv2.imshow("right", IMG_RIGHT)
    cv2.imshow("depth", disp)
    cv2.waitKey()
