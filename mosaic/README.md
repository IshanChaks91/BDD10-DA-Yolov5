# Mosaic Augmentation Implementation
Data Augmentation analysis of BDD100k (10K image set) in Yolov5

## Mosaic Data Augmentation:

Introduced as a technique to generate synthetic data in Yolov4, the mosaic data augmentation process involves the combination of 4 images into one image under certain ratios. Mosaic is an extension of another approach called CutMix.

## Approach:

Given location of directory containing 'train' images of BDD100k-10k Instance Segmentation dataset and the annotations file for this dataset, we perform synthetic image construction by category of the instance label in the annotation file. Final folder structure for the target directory looks like:

Directory_Location : '/home/.../Parent_Directory/'

```
Parent_directory
        |
        |_ _ _ train
        |         |_ _ _ images (mosaics per category, may not contain a single instance of the category)
        |         |         |_ _ _ car
        |         |         |_ _ _ person
        |         |         |_ _ _ truck
        |         |         |_ _ _ bus
        |         |         |_ _ _ bicycle
        |         |         |_ _ _ rider
        |         |         |_ _ _ trailer
        |         |         |_ _ _ motorcycle
        |         |         |_ _ _ caravan
        |         |         |_ _ _ train
        |         |         |_ _ _ actual_all (mosaics per catergory, only images with atleast one visible instance)
        |         |                  |_ _ _ car
        |         |                  |_ _ _ person
        |         |                  |_ _ _ truck
        |         |                  |_ _ _ bus
        |         |                  |_ _ _ bicycle
        |         |                  |_ _ _ rider
        |         |                  |_ _ _ trailer
        |         |                  |_ _ _ motorcycle
        |         |                  |_ _ _ caravan
        |         |                  |_ _ _ train
        |         |_ _ _ labels (mosaic labels per category, may not contain a single instance of the category)
        |                   |_ _ _ car
        |                   |_ _ _ person
        |                   |_ _ _ truck
        |                   |_ _ _ bus
        |                   |_ _ _ bicycle
        |                   |_ _ _ rider
        |                   |_ _ _ trailer
        |                   |_ _ _ motorcycle
        |                   |_ _ _ caravan
        |                   |_ _ _ train
        |                   |_ _ _ actual_all (mosaic labels per catergory, only images with atleast one visible instance)
        |                            |_ _ _ car
        |                            |_ _ _ person
        |                            |_ _ _ truck
        |                            |_ _ _ bus
        |                            |_ _ _ bicycle
        |                            |_ _ _ rider
        |                            |_ _ _ trailer
        |                            |_ _ _ motorcycle
        |                            |_ _ _ caravan
        |                            |_ _ _ train
        |
        |_ _ _ val
                  |_ _ _ images
                  |         |_ _ _ car
                  |         |_ _ _ person
                  |         |_ _ _ truck
                  |         |_ _ _ bus
                  |         |_ _ _ bicycle
                  |         |_ _ _ rider
                  |         |_ _ _ trailer
                  |         |_ _ _ motorcycle
                  |         |_ _ _ caravan
                  |         |_ _ _ train
                  |_ _ _ labels
                            |_ _ _ car
                            |_ _ _ person
                            |_ _ _ truck
                            |_ _ _ bus
                            |_ _ _ bicycle
                            |_ _ _ rider
                            |_ _ _ trailer
                            |_ _ _ motorcycle
                            |_ _ _ caravan
                            |_ _ _ train
```

Mosaic images are created on the basis of each category of instance. All images are processed to identify which category their instances belong to. This gives us bins for each category that consists of every image that contains atleast one instance of that category. Each bin is explored to randomly create atleast 1000 unique sets of 4 images. For each set, mosaic technique is applied to generate synthetic image.

Synthetic Image is created by concatenating 4 images into 2x2 grid with dimension (widthx2, heightx2), where width and height of each image is the same. 50% chance of harizontal flip and gamma correction is also applied to each image before mmosaic combination, to add more diversity in each instance occurence as well as imitate time of day for each image. In the dataset, each image is 1280x720. From the center of the grid, a threshold is described in both x and y axis in order to move the center to another point randomly within the threshold. With the new center selected, a mosaic 1280x720 is cropped. The threshold ensures each image in the grid is included in the mosaic under certain ratios. The threshold box is illustrated below:

```
         ______________________________________          ______________________________________
        |                  |                   |        |                  |                   |
        |                  |                   |        |                  |                   |
        |                  |                   |        |             _____|_____              |
        |                  |                   |        |            |     |     |             |
        |__________________x___________________|        |____________|_____x_____|_____________|
        |                  |                   |        |            |_____|_____|             |
        |                  |                   |        |                  |                   |
        |                  |                   |        |                  |                   | 
        |__________________|___________________|        |__________________|___________________|
```
A sample mosaic looks like:

![Figure 1: Sample Mosaic](/mosaic/img/0f172b7f-24d20001_00e9be89-00001570_3924f539-a72e7cc6_5f697884-f3f9d519.jpg)

In this approach, each mosaic which is created from a 4-image set of the same category, is then saved into .../train/images/'category'/ location. However, and the annotations for the mosaic are saved into .../train/labels/'category'/ location. Sampling criteria is as described: for each category, the total number of images where the instance appears atleast once is collected, shuffled and separated into 4 equal bins. Each set of 4 is then collected from each bin by order of the bin list. Shuffle can be repeated and new combinations can be made.

Note: In this method, after category based selection of 4-image set is obtained, applying the mosaic combination may not result in appearance of atleast a single occurence of the category instance in the frame, as mosaic frame view is randomly obtained from the grid frame within a certain threshold, which leaves out sections of the grid frame where the instance is located. Thus another directory is created 'actual_all' where category based images with atleast a single occurence is placed.

## Instruction:

1. Create environment (using environment.yaml / pyproject.toml / conda_requirements.txt / pip_requirements.txt) is not created earlier. Example:
```
conda create -n ENVNAME --file conda_requirements.txt
```
2. Open mosaic_path.yaml, update all location variables:
```
train_loc : <bdd100k_10k - train - images>
val_loc : <bdd100k_10k - val - images>
test_loc : <bdd100k_10k - test - images>
annotation_loc : < bdd100k_10k - labels - ins_seg - polygon - ins_seg_train.json>
mosaic_loc : < directory to save mosaic output>
```
3. Open mosaic.py, update location path to read mosaic_path.yaml file (line 10):
```
with open(".../././...", "r") as stream:
```
