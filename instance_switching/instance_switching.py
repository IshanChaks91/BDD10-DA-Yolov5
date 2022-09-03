#!/usr/bin/env python

from __future__ import division

import json
import yaml
import numpy as np
import is_utils
from shapely.geometry import Polygon

# Reading from is_path.yaml to store location of train and val images and location for IS augmentation
with open("/home/ichakr2s/my_projects/BDD10k_Data_Augmentaion/BDD10-DA-Yolov5/instance_switching/is_path.yaml", "r") as stream:
    try:
        is_path = yaml.safe_load(stream)
        train_location = str(is_path['train_loc'])
        val_location = str(is_path['val_loc'])
        is_location = str(is_path['is_loc'])
        annotation_location = str(is_path['annotation_loc'])
    except yaml.YAMLError as exc:
        print(exc)

img_height = 720
img_width = 1280
f_high = 1.55
f_low = 0.45
scale_factor = 0.0001
area_thresh = img_width * img_height * scale_factor

def main():
    for cat in bdd10k_category_list:
        if cat != 'car':
            file_names = []
            for idx in range(len(bdd_data)):
                for lab in bdd_data[idx]['labels']:
                    if lab['category'] == cat:
                        file_names.append(bdd_data[idx])
                        break

            shuf_idx = list(range(len(file_names)))
            np.random.shuffle(shuf_idx)
            if len(shuf_idx)%2 != 0:
                shuf_idx = shuf_idx[:-(len(shuf_idx)%2)]
            a_slice1 = shuf_idx[:int(len(shuf_idx)/2)]
            a_slice2 = shuf_idx[int(len(shuf_idx)/2):]

            for i in range(len(a_slice1)):
                img1 = file_names[a_slice1[i]]
                img2 = file_names[a_slice2[i]]
                idx_1_2 = []
                for idx1, annot1 in enumerate(img1['labels']):
                    if annot1['category'] == cat:
                        shape1 = np.array(annot1['poly2d'][0]['vertices'])
                        seg1 = Polygon(shape1)
                        if (np.max(shape1[:,0])-np.min(shape1[:,0])) > 0:
                            ann1_ratio = (np.max(shape1[:,1])-np.min(shape1[:,1]))/(np.max(shape1[:,0])-np.min(shape1[:,0]))
                            area1 = seg1.area
                            if area1 > area_thresh:
                                for idx2, annot2 in enumerate(img2['labels']):
                                    if annot2['category'] == cat:
                                        shape2 = np.array(annot2['poly2d'][0]['vertices'])
                                        seg2 = Polygon(shape2)
                                        if (np.max(shape2[:,0])-np.min(shape2[:,0])) > 0.: 
                                            ann2_ratio = (np.max(shape2[:,1])-np.min(shape2[:,1]))/(np.max(shape2[:,0])-np.min(shape2[:,0]))
                                            area2 = seg2.area
                                            if area2 > area_thresh:
                                                f = area1/area2
                                                if abs(ann2_ratio-ann1_ratio) < 1.:
                                                    if f < f_high and f > f_low:
                                                        idx_1_2.append([idx1, idx2])
                idx_1_2 = np.array(idx_1_2)
                try:
                    if len(idx_1_2) > 0:
                        for swap_idx in range(len(idx_1_2)):
                            ii2, ii1 = is_utils.swap(img1, img2, idx_1_2, swap_idx, train_location)
                            img_name2 = img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1]) + '__' + img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0])
                            is_utils.write_synth_img(img_name2, is_location, ii2, cat)
                            is_utils.make_label(img_name2, img2, bdd10k_category_list, img_width, img_height, is_location, cat)
                            img_name1 = img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0]) + '__' + img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1])
                            is_utils.write_synth_img(img_name1, is_location, ii1, cat)
                            is_utils.make_label(img_name1, img1, bdd10k_category_list, img_width, img_height, is_location, cat)
                except:
                    print('Warning while swapping.')

if __name__=="__main__":

    with open(annotation_location + 'ins_seg_train.json', 'r') as f:
        bdd_data = json.load(f)

    bdd10k_category_list = []
    for img in bdd_data:
        for properties in img['labels']:
            if properties['category'] not in bdd10k_category_list:
                bdd10k_category_list.append(properties['category'])

    # names string is used in the data.yaml file required to run Yolov5 model
    names = "'"
    for cat in bdd10k_category_list:
        names += cat + "', '" 

    # create directory location for augmented images
    for cat in bdd10k_category_list:

        is_utils.dir_category_create(is_location + 'train/images/', cat)
        is_utils.dir_category_create(is_location + 'train/labels/', cat)

        is_utils.dir_category_create(is_location + 'val/images/', cat)
        is_utils.dir_category_create(is_location + 'val/labels/', cat)

    # creates data.yaml, train and val location strings created in data.yaml file
    with open(is_location + 'data.yaml', 'w') as f:
        f.write('train: ' + train_location + 'images\n' + 'val: ' + val_location + 'images\n\n' + 'nc: ' + str(len(bdd10k_category_list)) + '\n' + 'names: [' + names[:-3] + ']')

    main()