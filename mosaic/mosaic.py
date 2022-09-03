from random import random
import mosaic_utils
import os
import yaml
import json
import time
import numpy as np

# Reading from mosaic_path.yaml to store location of train and val images and location for mosaic augmentation
with open("/home/ichakr2s/my_projects/BDD10k_Data_Augmentaion/BDD10-DA-Yolov5/mosaic/mosaic_path.yaml", "r") as stream:
    try:
        mosaic_path = yaml.safe_load(stream)
        train_location = str(mosaic_path['train_loc'])
        val_location = str(mosaic_path['val_loc'])
        mosaic_location = str(mosaic_path['mosaic_loc'])
        annotation_location = str(mosaic_path['annotation_loc'])
    except yaml.YAMLError as exc:
        print(exc)

def main():   

    # each category in category list of bdd100k_10k annotations
    for cat in category_bdd10:
        # ignore check for category==car to try to limit instances of car
        if cat != 'car':
            # create list of image annotations for the category
            category_image_array = []
            for img_idx in range(len(bdd_data)):
                for label in bdd_data[img_idx]['labels']:
                    if label['category'] == cat:
                        category_image_array.append(bdd_data[img_idx])
                        break
            
            # check number of mosaic images created in the category directory
            _, _, files = next(os.walk(mosaic_location + 'train/labels/' + cat))

            file_count = len(files)
            while file_count < 1000:

                # shuffle the images of the category, slice the total images into 4 bins
                # create 4 image set from one image of each bin with same index position

                category_array_idx = list(range(len(category_image_array)))
                np.random.shuffle(category_array_idx)
                
                # if size of category list is not divisible by 4, remove number of images
                #  from the list equal to reaminder
                if len(category_array_idx)%4 != 0:
                    category_array_idx = category_array_idx[:-(len(category_array_idx)%4)]

                category_array_idx_slice1 = category_array_idx[:int(len(category_array_idx)/4)]
                category_array_idx_slice2 = category_array_idx[int(len(category_array_idx)/4):int(len(category_array_idx)/2)]
                category_array_idx_slice3 = category_array_idx[int(len(category_array_idx)/2):int(len(category_array_idx)*3/4)]
                category_array_idx_slice4 = category_array_idx[int(len(category_array_idx)*3/4):]
                
                for indx in range(len(category_array_idx_slice1)):
                    im1_nm = category_image_array[category_array_idx_slice1[indx]]['name']
                    im2_nm = category_image_array[category_array_idx_slice2[indx]]['name']
                    im3_nm = category_image_array[category_array_idx_slice3[indx]]['name']
                    im4_nm = category_image_array[category_array_idx_slice4[indx]]['name']

                    # generate mosaic image by calling generate_mosaic() function
                    mosaic_utils.generate_mosaic(im1_nm, im2_nm, im3_nm, im4_nm, bdd_data, category_bdd10, cat, train_location, mosaic_location)

                # file count update
                _, _, files = next(os.walk(mosaic_location + 'train/labels/' + cat))
                file_count = len(files)

if __name__ == "__main__":

    # read bdd100k_10 train annotation file
    with open(annotation_location + 'ins_seg_train.json', 'r') as f:
        bdd_data = json.load(f)

    # create list of categories of bdd100k_10k
    category_bdd10 = []
    for img in bdd_data:
        for properties in img['labels']:
            if properties['category'] not in category_bdd10:
                category_bdd10.append(properties['category'])

    # names string is used in the data.yaml file required to run Yolov5 model
    names = "'"
    for cat in category_bdd10:
        names += cat + "', '" 

    # create directory location for augmented images
    for cat in category_bdd10:

        mosaic_utils.dir_category_create(mosaic_location + 'train/images/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'train/labels/', cat)

        mosaic_utils.dir_category_create(mosaic_location + 'val/images/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'val/labels/', cat)

        mosaic_utils.dir_category_create(mosaic_location + 'train/images/actual_all/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'train/labels/actual_all/', cat)  

    # creates data.yaml, train and val location strings created in data.yaml file
    with open(mosaic_location + 'data.yaml', 'w') as f:
        f.write('train: ' + train_location + 'images\n' + 'val: ' + val_location + 'images\n\n' + 'nc: ' + str(len(category_bdd10)) + '\n' + 'names: [' + names[:-3] + ']')


    # start = time.time()
    main()
    # end = time.time()
    # print(end-start)