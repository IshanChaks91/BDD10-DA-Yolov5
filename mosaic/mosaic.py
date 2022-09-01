from random import random
import mosaic_utils
import os
import yaml
import json
import time
import numpy as np

# Reading from mosaic_path.yaml to store location of train/val images
with open("/home/ic/Downloads/RnD_Augmentation/Mosaic/mosaic_path.yaml", "r") as stream:
    try:
        mosaic_path = yaml.safe_load(stream)
        train_location = str(mosaic_path['train_loc'])
        val_location = str(mosaic_path['val_loc'])
        mosaic_location = str(mosaic_path['mosaic_loc'])
        annotation_location = str(mosaic_path['annotation_loc'])
    except yaml.YAMLError as exc:
        print(exc)

def main():   

    for cat in category_bdd10:
        if cat != 'car':
            category_image_array = []
            for img_idx in range(len(bdd_data)):
                for label in bdd_data[img_idx]['labels']:
                    if label['category'] == cat:
                        category_image_array.append(bdd_data[img_idx])
                        break

            _, _, files = next(os.walk(mosaic_location + 'train/labels/' + cat))
            file_count = len(files)
            while file_count < 500:

                category_array_idx = list(range(len(category_image_array)))
                np.random.shuffle(category_array_idx)

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

                    mosaic_utils.generate_mosaic(im1_nm, im2_nm, im3_nm, im4_nm, bdd_data, category_bdd10, cat, train_location, mosaic_location)
                _, _, files = next(os.walk(mosaic_location + 'train/labels/' + cat))
                file_count = len(files)

if __name__ == "__main__":

    with open(annotation_location + 'ins_seg_train.json', 'r') as f:
        bdd_data = json.load(f)

    category_bdd10 = []
    for img in bdd_data:
        for properties in img['labels']:
            if properties['category'] not in category_bdd10:
                category_bdd10.append(properties['category'])
    names = "'"
    for cat in category_bdd10:
        names += cat + "', '" 

    for cat in category_bdd10:

        mosaic_utils.dir_category_create(mosaic_location + 'train/images/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'train/labels/', cat)

        mosaic_utils.dir_category_create(mosaic_location + 'val/images/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'val/labels/', cat)

        mosaic_utils.dir_category_create(mosaic_location + 'train/images/actual_all/', cat)
        mosaic_utils.dir_category_create(mosaic_location + 'train/labels/actual_all/', cat)  

    with open(mosaic_location + 'data.yaml', 'w') as f:
        f.write('train: ' + train_location + 'images\n' + 'val: ' + val_location + 'images\n\n' + 'nc: ' + str(len(category_bdd10)) + '\n' + 'names: [' + names[:-3] + ']')

    start = time.time()
    main()
    end = time.time()
    print(end-start)