import cv2
import random
import os
import numpy as np
from shapely.geometry import Polygon
import albumentations as A

# create directory and directory structure for mosaic data augmentation
def dir_category_create(mosaic_location, directory):
    """
        Function takes parent directory location where directory structure
        and child directories will be created.

        Parameters
        ----------
        mosaic_location : location of parent directory from mosaic_path.yaml <type:str>
        directory : child directory name as per category <type:str> 
        
        Returns
        -------
        Raises
        ------
    """
    if not os.path.exists(mosaic_location + directory):
        os.makedirs(mosaic_location + directory)

# create function to clip bounding box that moves outside image boundary
def clip(minx, miny, maxx, maxy, width, height):

    """
        Function takes the bounding box parameters of an instance 
        in an image and clips the corners that exist outside the 
        image frame. Clipping Applies to partially visible bounding boxes 
        and invisible bounding boxes.

        Parameters
        ----------
        minx : min x value of bounding box <type:int>
        miny : min y value of bounding box <type:int>
        maxx : max x value of bounding box <type:int>
        maxy : max y value of bounding box <type:int>
        width : width of the image file in pixels <type:int>
        height : height of the image file  in pixels <type:int>
        
        Returns
        -------
        Boolean : indicates whether bounding box exists within image frame
        List : updated bounding box dimensions

        Raises
        ------
    """

    if (minx > 0 and minx < width) or (maxx > 0 and maxx < width):
        if minx < 0:
            minx = 0
        if maxx > width:
            maxx = width
        if (miny > 0 and miny < height) or (maxy > 0 and maxy < height):
            if miny < 0:
                miny = 0
            if maxy > height:
                maxy = height
            return True, [minx, miny, maxx, maxy]
        else:
            return False, []
    else:
        return False, []


def generate_mosaic(im1_nm, im2_nm, im3_nm, im4_nm, bdd_data, category_bdd10, category_type, train_location, mosaic_location):

    """
        Function takes 4 image file names from the BBD100k-10k 'train' 
        dataset for instance segmentation, a list which contains the 
        annotations for the 'train' dataset, list of categories in the 
        dataset, category type, training image location, and directory 
        location in which the synthetic images and labels are saved.
        Creates 2x2 grid based mosaic of 4 images of respective 
        catergory_type. 
        
        The images are selected as per category in the dataset:
        
            category types: ['car', 'person', 'truck', 'bus', 'bicycle', 'rider', 'trailer', 'motorcycle', 'caravan', 'train']

        Parameters
        ----------
        im1_nm : name of image file <type:str>
        im2_nm : name of image file <type:str>
        im3_nm : name of image file <type:str>
        im4_nm : name of image file <type:str>
        bdd_data : list of annotations for the training dataset <type:json>
        train_location : location of training images dataset <type:str>
        mosaic_location : location to save the synthetic images <type:str>
        
        Returns
        -------
        creates in location mosaic of the 4 images as per category, also creates 
        directory for mosaic images that contain instances per catergory creates 
        in location annotaion of the mosaic as per yolov5pytorch txt format for yolov5
        
        Raises
        ------
    """

    # variables to check horizontal flip
    check_flip1 = False
    check_flip2 = False
    check_flip3 = False
    check_flip4 = False

    # list to store value between 0.7 and 3.2, a random value selected from list to feed
    # albumentation gamma function applied on image.
    g_val = np.arange(0.7, 3.2, 0.3).tolist()
    
    # albumentation horizontal flip funtion (p=1 applies 100% probability of flip)
    transform_horizontal = A.HorizontalFlip(p=1.)

    # string to capture image and label - file name format, to save image and annotation of mosaic
    mosaic_file_name = im1_nm[:-4] + '_' + im2_nm[:-4] + '_' + im3_nm[:-4] + '_' + im4_nm[:-4]

    # Maximum values of x and y of the tolerance box around the center of the grid frame,
    # within which the center of the mosaic frame is bounded to ensure each image of the
    # grid frame is taken in the mosaic frame

    x_tol = 200
    y_tol = 100

    # Reading lables of each image annotation from the the annotations file (bdd_data) 
    # into variables to be used to read image from image source directory
    for im in bdd_data:
        if im['name'] == im1_nm:
            lb1 = im
        if im['name'] == im2_nm:
            lb2 = im
        if im['name'] == im3_nm:
            lb3 = im
        if im['name'] == im4_nm:
            lb4 = im

    # read and save images 
    im1 = cv2.imread(train_location + lb1['name'])
    im2 = cv2.imread(train_location + lb2['name'])
    im3 = cv2.imread(train_location + lb3['name'])
    im4 = cv2.imread(train_location + lb4['name'])

    # for each image, apply 50% chance of gamma transform
    if random.random() > 0.5:
        gamma_val = random.sample(g_val, k=1)[0]
        im1 = A.gamma_transform(im1, gamma=gamma_val)
    #for each image, apply 50% chance of horizontal flip transform
    # create flag if flip occurs 
    if random.random() > 0.5:
        im1 = transform_horizontal(image=im1)['image']
        check_flip1 = True
    
    if random.random() > 0.5:
        gamma_val = random.sample(g_val, k=1)[0]
        im2 = A.gamma_transform(im2, gamma=gamma_val)
    if random.random() > 0.5:
        im2 = transform_horizontal(image=im2)['image']
        check_flip2 = True

    if random.random() > 0.5:
        gamma_val = random.sample(g_val, k=1)[0]
        im3 = A.gamma_transform(im3, gamma=gamma_val)
    if random.random() > 0.5:
        im3 = transform_horizontal(image=im3)['image']
        check_flip3 = True

    if random.random() > 0.5:
        gamma_val = random.sample(g_val, k=1)[0]
        im4 = A.gamma_transform(im4, gamma=gamma_val)
    if random.random() > 0.5:
        im4 = transform_horizontal(image=im4)['image']
        check_flip4 = True

    # vertical concat images 1 and 2, and 3 and 4. Follow with horizontal 
    # concat of the two sets into im_stacked. This is the grid frame of the mossaic
    im_v1 = cv2.vconcat([im1, im2])
    im_v2 = cv2.vconcat([im3, im4])
    im_stacked = cv2.hconcat([im_v1, im_v2])

    # find the center of the grid frame
    x_cen = np.shape(im_stacked)[1] / 2
    y_cen = np.shape(im_stacked)[0] / 2

    # create new center ker_cen (kernel center) for mosaic frame within the tolerance box 
    ker_cen = [y_cen + np.random.randint(-y_tol, y_tol), x_cen + np.random.randint(-x_tol, x_tol)]
    # create mosaic frame by slicing the grid frame from ker_cen
    mosaic = im_stacked[int(ker_cen[0]-(y_cen/2)):int(ker_cen[0]+(y_cen/2)), int(ker_cen[1]-(x_cen/2)):int(ker_cen[1]+(x_cen/2))]

    # create mosaic image
    cv2.imwrite(mosaic_location + 'train/images/' + category_type + '/' + mosaic_file_name + '.jpg', mosaic)

    # create string to capture annotation of each instance of each of the 4 images in Yolov5 format
    # Format: class_id center_x center_y width height
    yolostr = ''

    # transforming the annotations of the 4 images with respect to the mosaic frame reference
    # and using the clip() function to update the bounding boxes

    # size of the mosaic image for the bdd100k-10k dataset is 1280x720, which is equal to center of grid frame
    mosaic_width = x_cen
    mosaic_height = y_cen

    # for each stacked image in the grid frame (im1.im2.im3.im4)

    # flag to track whether category is present inside mosaic frame
    check_cat = False
    # use the annotations of each label in the image to copy modified vertices of labels to string
    for an in lb1['labels']:
        # change the reference of each pixel from the grid origin
        ver = np.array(an['poly2d'][0]['vertices'])
        if check_flip1:
            # update bounding box location of each label in flipped image
            ver = np.vstack((1280 - ver[:,0] , ver[:,1])).T
        vert = ver - np.array([ker_cen[1]-mosaic_width/2, ker_cen[0]-mosaic_height/2])
        # create mask of instance using ins seg vertices from the annotations
        segment = Polygon(vert)
        # find the bounds of the vertices
        minx, miny, maxx, maxy = segment.bounds
        # clip the bounding boxes of the instances that are outside the mosaic frame
        flag, i_bds = clip(minx, miny, maxx, maxy, mosaic_width, mosaic_height)
        if flag:
            wth, hgt = (i_bds[2] - i_bds[0])/mosaic_width, (i_bds[3] - i_bds[1])/mosaic_height
            centr_x = ((i_bds[2] + i_bds[0])/2)/mosaic_width
            centr_y = ((i_bds[3] + i_bds[1])/2)/mosaic_height
            cat = category_bdd10.index(an['category'])
            # if the category which appears in the mosaic frame is the category type then
            if an['category'] == category_type:
                check_cat = True
            # create yolov5pytorch txt format annotation for yolov5
            yolostr += str(cat) + ' ' + str(centr_x) + ' ' + str(centr_y) + ' ' + str(wth) + ' ' + str(hgt)  + '\n'

    for an in lb2['labels']:
        # change the reference of each pixel from the grid origin mosaic frame origin, 
        # lb2 is stacked vertically below lb1 so there needs to be an offset of height of image (720)
        # for each pixel position
        ver = np.array(an['poly2d'][0]['vertices'])
        if check_flip2:
            ver = np.vstack((1280 - ver[:,0] , ver[:,1])).T
        vert = ver + np.array([0, mosaic_height]) - np.array([ker_cen[1]-mosaic_width/2, ker_cen[0]-mosaic_height/2])
        segment = Polygon(vert)
        minx, miny, maxx, maxy = segment.bounds
        flag, i_bds = clip(minx, miny, maxx, maxy, mosaic_width, mosaic_height)
        if flag:
            wth, hgt = (i_bds[2] - i_bds[0])/mosaic_width, (i_bds[3] - i_bds[1])/mosaic_height
            centr_x = ((i_bds[2] + i_bds[0])/2)/mosaic_width
            centr_y = ((i_bds[3] + i_bds[1])/2)/mosaic_height
            cat = category_bdd10.index(an['category'])
            if an['category'] == category_type:
                check_cat = True
            yolostr += str(cat) + ' ' + str(centr_x) + ' ' + str(centr_y) + ' ' + str(wth) + ' ' + str(hgt)  + '\n'

    for an in lb3['labels']:
        # change the reference of each pixel from the grid origin mosaic frame origin, 
        # lb3 is stacked horizontlly beside lb1 so there needs to be an offset of width of image (1280)
        # for each pixel position
        ver = np.array(an['poly2d'][0]['vertices'])
        if check_flip3:
            ver = np.vstack((1280 - ver[:,0] , ver[:,1])).T
        vert = ver + np.array([x_cen, 0]) - np.array([ker_cen[1]-mosaic_width/2, ker_cen[0]-mosaic_height/2])
        segment = Polygon(vert)
        minx, miny, maxx, maxy = segment.bounds
        flag, i_bds = clip(minx, miny, maxx, maxy, mosaic_width, mosaic_height)
        if flag:
            wth, hgt = (i_bds[2] - i_bds[0])/mosaic_width, (i_bds[3] - i_bds[1])/mosaic_height
            centr_x = ((i_bds[2] + i_bds[0])/2)/mosaic_width
            centr_y = ((i_bds[3] + i_bds[1])/2)/mosaic_height
            cat = category_bdd10.index(an['category'])
            if an['category'] == category_type:
                check_cat = True
            yolostr += str(cat) + ' ' + str(centr_x) + ' ' + str(centr_y) + ' ' + str(wth) + ' ' + str(hgt)  + '\n'

    for an in lb4['labels']:
        # change the reference of each pixel from the grid origin mosaic frame origin, 
        # lb4 is stacked vertically below lb3 so there needs to be an offset of width and height of image (1280,720)
        # for each pixel position
        ver = np.array(an['poly2d'][0]['vertices'])
        if check_flip4:
            ver = np.vstack((1280 - ver[:,0] , ver[:,1])).T
        vert = ver + np.array([mosaic_width, mosaic_height]) - np.array([ker_cen[1]-mosaic_width/2, ker_cen[0]-mosaic_height/2])
        segment = Polygon(vert)
        minx, miny, maxx, maxy = segment.bounds
        flag, i_bds = clip(minx, miny, maxx, maxy, mosaic_width, mosaic_height)
        if flag:
            wth, hgt = (i_bds[2] - i_bds[0])/mosaic_width, (i_bds[3] - i_bds[1])/mosaic_height
            centr_x = ((i_bds[2] + i_bds[0])/2)/mosaic_width
            centr_y = ((i_bds[3] + i_bds[1])/2)/mosaic_height
            cat = category_bdd10.index(an['category'])
            if an['category'] == category_type:
                check_cat = True
            yolostr += str(cat) + ' ' + str(centr_x) + ' ' + str(centr_y) + ' ' + str(wth) + ' ' + str(hgt)  + '\n'

    # create annotation txt file for the mosaic image
    with open(mosaic_location + 'train/labels/' + category_type + '/'  + mosaic_file_name + '.txt' , 'w') as f:
        f.write(yolostr)

    # if condition then take the mosaic image and its annotations and assemble in separate location, applies
    # to images where mosaic includes instance of category
    if check_cat:
        cv2.imwrite(mosaic_location + 'train/images/actual_all/' + category_type + '/' + mosaic_file_name + '.jpg', mosaic)
        with open(mosaic_location + 'train/labels/actual_all/' + category_type + '/'  + mosaic_file_name + '.txt' , 'w') as f:
            f.write(yolostr)