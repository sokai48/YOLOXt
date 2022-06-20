"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import os

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset_dir = '/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/kitti_4class/'
    train_file = open(os.path.join(dataset_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(dataset_dir, 'val.txt'), 'w')
    file_ids = ["%06d" % i for i in range(0, 6480)]
    train_ids, val_ids = train_test_split(file_ids, test_size=0.2)

    for ids in train_ids :
        ids = "images/" + ids + ".png"
        ids = dataset_dir + ids + "\n"
        print(ids)
        train_file.write(ids)

    train_file.close()

    for ids in val_ids :
        ids = "images/" + ids + ".png"
        ids = dataset_dir + ids + "\n"
        val_file.write(ids)

    val_file.close()

   

