#!/usr/bin/env python

import os
import cv2
import numpy as np
from shapely.geometry import Polygon

# create directory and directory structure for IS data augmentation
def dir_category_create(is_location, directory):
    """
        Function takes parent directory location where child directories 
        will be created.

        Parameters
        ----------
        is_location : location of parent directory from is_path.yaml <type:str>
        directory : child directory name as per category <type:str> 
        
        Returns
        -------
        Raises
        ------
    """
    if not os.path.exists(is_location + directory):
        os.makedirs(is_location + directory)

# clip normalized parameters in yolov5pytorchtxt file (required for training)
def clip(param):
    """
        Function takes parameter value and clips it to 0.9999 if greater than 1.0

        Parameters
        ----------
        param: value to clip <type:float> 
        
        Returns
        -------
        param: clipped value <= 1.0 <type:float>

        Raises
        ------
    """
    if param > 1.0:
        return 0.9999
    else:
        return param

# create image in location
def write_synth_img(img_name, output_location, image, category):
    """
        Function takes image name, output location, image and instance category
        to create image in location with tailored name

        Parameters
        ----------
        img_name: name for the image <type:str>
        output_location: directory location to save image <type:str>
        image: image to be saved <type:str>
        category: category of the instance <type:str>
        
        Returns
        -------
        image file

        Raises
        ------
    """
    cv2.imwrite(output_location + 'train/images/' + category + '/' + img_name + '.jpg', image)

def make_label(file_name_str, train_img, bdd10k_category_list, img_width, img_height, output_location, category):
    """
        Function takes file name, image, category_list, image width, 
        image height, output location, category to make label txt file 
        in yolov5pytorchtxt format

        Parameters
        ----------
        file_name_str: file name <type:str>
        train_img: image json <type:json>
        bdd10k_category_list: category list <type:list>
        img_width: width of image <type:int>
        img_height: height of image <type:int>
        output_location: parent directory location to save label <type:str>
        category: category of instance <type:str>
        
        Returns
        -------
        .txt label file

        Raises
        ------
    """
    strinG = ''
    # each instance annotation in image file
    for label in train_img['labels']:
        # polygon object of the object
        poly = Polygon(label['poly2d'][0]['vertices'])
        # category index number of the instance label as per the category list
        cat = bdd10k_category_list.index(label['category'])
        # yolov5pytorchtxt format based label name <class_id center_x center_y width height>
        # where fields are space delimited, and the coordinates are normalized from zero to one 
        centr_xy =  poly.centroid.xy
        centr_x = centr_xy[0][0]/img_width
        centr_y = centr_xy[1][0]/img_height
        minx, miny, maxx, maxy = poly.bounds
        width, height = (maxx - minx)/img_width, (maxy - miny)/img_height
        strinG += str(cat) + ' ' + str(clip(centr_x)) + ' ' + str(clip(centr_y)) + ' ' + str(clip(width)) + ' ' + str(clip(height))  + '\n'
    with open(output_location + 'train/labels/' + category + '/' + file_name_str + '.txt' , 'w') as f:
        f.write(strinG)

def swap(i1, i2, swap_idx, train_location):
    """
        Function takes annotation of image pair i1 and i2, along with swappable indices 
        of i1 and i2 and train location

        Parameters
        ----------
        i1: image 1 annotations <type:json>
        i2: image 2 annotations <type:json>
        swap_idx: list of indices to swap <type:list>
        train_location: location of train images <type:str>
        
        Returns
        -------
        swapped_1: 
        swapped_2: 

        Raises
        ------
    """

    # read image 
    image1 = cv2.imread(train_location + i1['name'])
    # extract annotations of image 1 and annotation of image 1 which must be swapped
    annot1 = i1['labels'][swap_idx[0]]
    # extract vertices of instance
    roi_corners1 = np.array([annot1['poly2d'][0]['vertices']], dtype=np.int32)
    # extract minx, miny, maxx, maxy of the vertices
    xmin1 = np.min(roi_corners1[0][:,0])
    xmax1 = np.max(roi_corners1[0][:,0])
    ymin1 = np.min(roi_corners1[0][:,1])
    ymax1 = np.max(roi_corners1[0][:,1])
    # 
    roi_mean1 = np.array([(xmax1+xmin1)/2, (ymax1+ymin1)/2])
    roi_norm1 = roi_corners1[0] - roi_mean1
    
    # read image 
    image2 = cv2.imread(train_location + i2['name'])
    # extract annotations of image 2 and annotation of image 2 which must be swapped
    annot2 = i2['labels'][swap_idx[1]]
    # extract vertices of instance
    roi_corners2 = np.array([annot2['poly2d'][0]['vertices']], dtype=np.int32)
    # extract minx, miny, maxx, maxy of the vertices
    xmin2 = np.min(roi_corners2[0][:,0])
    xmax2 = np.max(roi_corners2[0][:,0])
    ymin2 = np.min(roi_corners2[0][:,1])
    ymax2 = np.max(roi_corners2[0][:,1])
    roi_mean2 = np.array([(xmax2+xmin2)/2, (ymax2+ymin2)/2])
    roi_norm2 = roi_corners2[0] - roi_mean2

    mask1 = np.ones(image1.shape, dtype=np.uint8)
    mask1.fill(255)
    cv2.fillPoly(mask1, roi_corners1, 0)
    masked_image1 = cv2.bitwise_or(image1, mask1)    
    masking_obj1 = masked_image1[ymin1:ymax1, xmin1:xmax1]    
    masked_image1 = np.ones(image2.shape, dtype=np.uint8)
    masked_image1.fill(255)
    masking_obj1 = cv2.resize(masking_obj1, (xmax2-xmin2, ymax2-ymin2), interpolation = cv2.INTER_AREA)
    masked_image1[ymin2:ymax2, xmin2:xmax2] = masking_obj1
    img2_mask = np.vstack((((xmax2-xmin2)/float(xmax1-xmin1))*roi_norm1[:,0], ((ymax2-ymin2)/float(ymax1-ymin1))*roi_norm1[:,1])).T
    mask2 = np.ones(image2.shape, dtype=np.uint8)
    mask2.fill(255)
    cv2.fillPoly(mask2, np.array([img2_mask+roi_mean2], dtype=np.int), 0)   
    mask2 = 255 - mask2
    mask2 = cv2.blur(mask2,(5,5))
    masked_image2 = cv2.bitwise_or(image2, mask2)
    swapped_1 = cv2.bitwise_and(masked_image1, masked_image2)
    
    mask2 = np.ones(image2.shape, dtype=np.uint8)
    mask2.fill(255)
    cv2.fillPoly(mask2, roi_corners2, 0)
    masked_image2 = cv2.bitwise_or(image2, mask2)
    masking_obj2 = masked_image2[ymin2:ymax2, xmin2:xmax2]
    masked_image2 = np.ones(image1.shape, dtype=np.uint8)
    masked_image2.fill(255)
    masking_obj2 = cv2.resize(masking_obj2, (xmax1-xmin1, ymax1-ymin1), interpolation = cv2.INTER_AREA)
    masked_image2[ymin1:ymax1, xmin1:xmax1] = masking_obj2
    img1_mask = np.vstack((((xmax1-xmin1)/float(xmax2-xmin2))*roi_norm2[:,0], ((ymax1-ymin1)/float(ymax2-ymin2))*roi_norm2[:,1])).T
    mask1 = np.ones(image1.shape, dtype=np.uint8)
    mask1.fill(255)
    cv2.fillPoly(mask1, np.array([img1_mask+roi_mean1], dtype=np.int), 0)
    mask1 = 255 - mask1
    mask1 = cv2.blur(mask1,(5,5))
    masked_image1 = cv2.bitwise_or(image1, mask1)
    swapped_2 = cv2.bitwise_and(masked_image2, masked_image1)
    
    return swapped_1, swapped_2