 # -*- coding: utf-8 -*-
import Augmentor
import glob
import random
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_path = 'data/img'
groud_truth_path = 'data/mask'
tmpRoot = 'tmp/'
img_type = 'jpg'

def doAugment():
    img = glob.glob('tmp/data/img/*.' + img_type)
    masks = glob.glob('tmp/data/mask/*.' + img_type)
    sum = 0

    p = Augmentor.Pipeline(tmpRoot + train_path)
    p.ground_truth(tmpRoot + groud_truth_path)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)  # 旋转
    p.rotate90(probability=0.2)
    p.rotate180(probability=0.2)
    p.rotate270(probability=0.2)
    p.flip_left_right(probability=0.5)  # 按概率左右翻转
    p.flip_top_bottom(probability=0.5)  # 按概率随即上下翻转
    p.sample(500)

    # for img in train_img:
    #     p = Augmentor.Pipeline(train_tmp_path+'/'+str(i))
    #     p.ground_truth(mask_tmp_path+'/'+str(i))
    #     p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)#旋转
    #     p.flip_left_right(probability=0.5)#按概率左右翻转
    #     p.zoom_random(probability=0.6, percentage_area=0.99)#随即将一定比例面积的图形放大至全图
    #     p.flip_top_bottom(probability=0.6)#按概率随即上下翻转
    #     count = random.randint(1, 3)
    #     print("\nNo.%s data is being augmented and %s data will be created"%(i,count))
    #     sum = sum + count
    #     p.sample(count)
    #     print("Done")
    # print("%s pairs of data has been created totally"%sum)


doAugment()

