# Instance Switching Augmentation Implementation
Data Augmentation analysis of BDD100k (10K image set) in Yolov5

## Instance Switching Data Augmentation:

Introduced as a technique to generate synthetic data in Yolov4, the instance switching data augmentation process involves the pairing of valid images - where atleast one instance of a particular class in each image of the pair can become a candidate for switching between the two images. Candidacy for switching is determined by factoring in the size and shape of the two instances, size threshhold compared to image etc.

## Approach:

Given location of directory containing 'train' images of BDD100k-10k Instance Segmentation dataset and the annotations file for this dataset, we perform synthetic image construction by category of the instance label in the annotation file. Training is done on Yolov5 model, hence annotation format used is yolov5pytorchtxt. Final folder structure for the target directory looks like:

Directory_Location : '/home/.../Parent_Directory/'

```
Parent_directory
        |
        |_ _ _ train
        |         |_ _ _ images
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
        |         |_ _ _ labels
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

Instance Switching images are created on the basis of each category of instance. All images are processed to identify which category their instances belong to. This gives us bins for each category that consists of every image that contains atleast one instance of that category. All images for a category are divided equally into two lists. A pair of images is created by taking two images from the two lists by order of index.

Inside the pair, all instances for the category/class from each image are paired to check candidacy criteria. When candidate pair of instances is found we switch the instances between the two images, this creates 2 new images from the pair of images we started off with earlier. This image pairing and candidate instance pair selection process continues until are aiamge pairs are exhausted from the two image lists and the nwe move on to the next category.

A sample instance switching looks like:

![Figure 1: Sample instance_switching](/instance_switching/img/00e9be89-00001430_0__2a92cf41-00000000_0.jpg)
![Figure 2: Sample instance_switching](/instance_switching/img/2a92cf41-00000000_0__00e9be89-00001430_0.jpg)

In this approach, each instance switching synthetic pair which is created from a 2-image set of the same category, is then saved into .../train/images/'category'/ location. Moreover, and the annotations for the instance swithcing synthetic pair are saved into .../train/labels/'category'/ location. Sampling criteria is as described: for each category, the total number of images where the instance appears atleast once is collected, shuffled and separated into 2 equal bins. Each set of 2 is then collected from each bin by order of the bin list index. Shuffle can be repeated and new combinations can be made.

Note: In this method, after category based selection of 2-image set is obtained, applying the instance switching depends on our thresholding criteria.

## Instruction:

1. Create environment (using environment.yaml / pyproject.toml / conda_requirements.txt / pip_requirements.txt) from the conda_env folder if not created earlier. Example:
```
conda create -n ENVNAME --file conda_requirements.txt
```
2. Open is_path.yaml, update all location variables:
```
train_loc : <bdd100k_10k - train - images>
val_loc : <bdd100k_10k - val - images>
test_loc : <bdd100k_10k - test - images>
annotation_loc : < bdd100k_10k - labels - ins_seg - polygon - ins_seg_train.json>
mosaic_loc : < directory to save instance_switching output>
```
3. Open instance_switching.py, update location path to read is_path.yaml file (line 12):
```
with open(".../././...", "r") as stream:
```
4. Run instance_switching.py
