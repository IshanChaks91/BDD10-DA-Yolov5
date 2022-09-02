#!/usr/bin/env python

import json
import numpy as np
import utils
from shapely.geometry import Polygon

dir_location = '/home/ic/Downloads/RnD_Augmentation/bdd100k_ins_seg_labels_trainval/bdd100k/labels/ins_seg/polygons/'
train_location = '/home/ic/Downloads/RnD_Augmentation/bdd100k_images_10k/bdd100k/images/10k/train/'
output_location = '/home/ic/Downloads/RnD_Augmentation/IS_New/IS_Aug/'

img_height = 720
img_width = 1280
f_high = 1.55
f_low = 0.45
scale_factor = 0.001
area_thresh = img_width * img_height * scale_factor

with open(dir_location + 'ins_seg_train.json', 'r') as f:
    bdd_data = json.load(f)

bdd10k_category_list = []
for img in bdd_data:
  for properties in img['labels']:
    if properties['category'] not in bdd10k_category_list:
      bdd10k_category_list.append(properties['category'])

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
                        ann1_ratio = (np.max(shape1[:,1])-np.min(shape1[:,1]))/(np.max(shape1[:,0])-np.min(shape1[:,0]))
                        area1 = seg1.area
                        if area1 > area_thresh:
                            for idx2, annot2 in enumerate(img2['labels']):
                                if annot2['category'] == cat:
                                    shape2 = np.array(annot2['poly2d'][0]['vertices'])
                                    seg2 = Polygon(shape2)
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
                            ii2, ii1 = utils.swap(img1, img2, idx_1_2, swap_idx, train_location)
                            img_name2 = img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1]) + '__' + img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0])
                            utils.write_synth_img(img_name2, output_location, ii2)
                            utils.make_label(img_name2, img2, bdd10k_category_list, img_width, img_height, output_location)
                            img_name1 = img1['name'][:-4] + '_' + str(idx_1_2[swap_idx][0]) + '__' + img2['name'][:-4] + '_' + str(idx_1_2[swap_idx][1])
                            utils.write_synth_img(img_name1, output_location, ii1)
                            utils.make_label(img_name1, img1, bdd10k_category_list, img_width, img_height, output_location)
                except ValueError:
                    print('Error broadcasting shapes while swapping.')

if __name__=="__main__":
    main()