"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import os

from sklearn.model_selection import train_test_split
import shutil

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):
    image_names = []
    only_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
                only_names.append(filename.split(".")[0])
    return image_names, only_names

if __name__ == '__main__':




    dataset_dir = '/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/newbdd_2/'

    '''

    img_dir = '/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/newbdd_2/images/'

    image_names, only_names = get_image_list(img_dir)

    i = 0 

    only_names.sort()
    
    for name in only_names :

        i += 1
        print("%06d" % i)
        os.rename( dataset_dir + "images/" + name + ".jpg" , dataset_dir + "images/" + "%06d" % i + ".jpg" )
        os.rename( dataset_dir + "labels/"+ name+ ".txt", dataset_dir + "labels/" + "%06d" % i + ".txt")
        os.rename( dataset_dir  + "roadgt/"+ name+".png"  ,dataset_dir + "roadgt/"+"%06d" % i + ".png")

    '''



    train_file = open(os.path.join(dataset_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(dataset_dir, 'val.txt'), 'w')
    file_ids = ["%06d" % i for i in range(1, 6109)]
    train_ids, val_ids = train_test_split(file_ids, test_size=0.2)

    for ids in train_ids :
        ids = "images/" + ids + ".jpg"
        
        ids = dataset_dir + ids + "\n"
        print(ids)
        train_file.write(ids)
        

    train_file.close()

    for ids in val_ids :
        ids = "images/" + ids + ".jpg"
        
        ids = dataset_dir + ids + "\n"
        val_file.write(ids)

    val_file.close()


