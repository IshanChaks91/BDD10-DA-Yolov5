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

# ratio of area limit between f_high and f_low, for candidate pair selection
f_high = 1.55
f_low = 0.45
# instance area greater than threshold for each candidate instance selection
scale_factor = 0.0001
area_thresh = img_width * img_height * scale_factor

def main():
    # for each category (!=car) in the instance category list
    for cat in bdd10k_category_list:
        if cat != 'car':
            # find all images for this category (image contains all of its annotation data)
            file_names = []
            for idx in range(len(bdd_data)):
                for lab in bdd_data[idx]['labels']:
                    if lab['category'] == cat:
                        file_names.append(bdd_data[idx])
                        break
            # suffle the indices of the image list
            shuf_idx = list(range(len(file_names)))
            np.random.shuffle(shuf_idx)
            # split the list of indices into 2 halves
            if len(shuf_idx)%2 != 0:
                shuf_idx = shuf_idx[:-(len(shuf_idx)%2)]
            a_slice1 = shuf_idx[:int(len(shuf_idx)/2)]
            a_slice2 = shuf_idx[int(len(shuf_idx)/2):]

            # create pairs with an image from each list containing list of indices of image names
            for i in range(len(a_slice1)):
                img1 = file_names[a_slice1[i]]
                img2 = file_names[a_slice2[i]]
                idx_1_2 = []
                # for each image in the pair
                for idx1, annot1 in enumerate(img1['labels']):
                    # check category of each instance
                    if annot1['category'] == cat:
                        # extract vertices of the instance
                        shape1 = np.array(annot1['poly2d'][0]['vertices'])
                        # create Polygon object of first instance
                        seg1 = Polygon(shape1)
                        # check instance height is != 0
                        if (np.max(shape1[:,0])-np.min(shape1[:,0])) > 0:
                            # max height / max width of the instance
                            ann1_ratio = (np.max(shape1[:,1])-np.min(shape1[:,1]))/(np.max(shape1[:,0])-np.min(shape1[:,0]))
                            # instance area
                            area1 = seg1.area
                            # check instance area is greater than area threshold
                            if area1 > area_thresh:
                                # compare instance with all instances of other image of candidate pair
                                for idx2, annot2 in enumerate(img2['labels']):
                                    if annot2['category'] == cat:
                                        shape2 = np.array(annot2['poly2d'][0]['vertices'])
                                        seg2 = Polygon(shape2)
                                        if (np.max(shape2[:,0])-np.min(shape2[:,0])) > 0.: 
                                            ann2_ratio = (np.max(shape2[:,1])-np.min(shape2[:,1]))/(np.max(shape2[:,0])-np.min(shape2[:,0]))
                                            area2 = seg2.area
                                            if area2 > area_thresh:
                                                # calculate ratio of areas of both instances
                                                f = area1/area2
                                                # bounding box of both instances have similar height to width ratio withing threshold
                                                if abs(ann2_ratio-ann1_ratio) < 1.:
                                                    # area ratio of each instance is within threshold 
                                                    if f < f_high and f > f_low:
                                                        # pair is selected as a candidate for instance switching
                                                        idx_1_2.append([idx1, idx2])
                idx_1_2 = np.array(idx_1_2)
                try:
                    # generate 2 images and 2 image labels for each candidate instance pair of the category by reading the candidate instance pair index list idx_1_2
                    if len(idx_1_2) > 0:
                        for swap_idx in idx_1_2:
                            # generate two images based on candidate instance pair indices in swap_idx
                            ii2, ii1 = is_utils.swap(img1, img2, swap_idx, train_location)
                            img_name2 = img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1]) + '__' + img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0])
                            # create image in location is_location
                            is_utils.write_synth_img(img_name2, is_location, ii2, cat)
                            # create label of image in label location
                            is_utils.make_label(img_name2, img2, bdd10k_category_list, img_width, img_height, is_location, cat)
                            img_name1 = img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0]) + '__' + img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1])
                            is_utils.write_synth_img(img_name1, is_location, ii1, cat)
                            is_utils.make_label(img_name1, img1, bdd10k_category_list, img_width, img_height, is_location, cat)
                except:
                    print('Warning while swapping.')

if __name__=="__main__":
    # read annotation json fro train images
    with open(annotation_location + 'ins_seg_train.json', 'r') as f:
        bdd_data = json.load(f)
    # category list of all instances
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