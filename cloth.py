import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
from matplotlib.path import Path
import json
from os.path import exists
import pandas as pd
# os.chdir('/content/drive/MyDrive/CS543-FinalProject')
import person_segmentation.person_sgg
import utils_deepfashion
from google.colab.patches import cv2_imshow
import evaluation_metrics
import visualize
import nms
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import copy



deepfashion_dict_grams= {}

deepfashion_dict_grams["short sleeve top"]= 160
deepfashion_dict_grams["long sleeve top"] = 230
deepfashion_dict_grams["short sleeve outwear"] = 490 
deepfashion_dict_grams["long sleeve outwear"] = 590
deepfashion_dict_grams["vest"] = 150
deepfashion_dict_grams["sling"] = 300
deepfashion_dict_grams["shorts"] = 260
deepfashion_dict_grams["trousers"] = 550 
deepfashion_dict_grams["skirt"] = 250
deepfashion_dict_grams["short sleeve dress"] = 350
deepfashion_dict_grams["long sleeve dress"] = 450
deepfashion_dict_grams["vest dress"] = 320
deepfashion_dict_grams["sling dress"]= 400


def get_cloth(cloth_items=None, cloth_areas=None, A_cov0=None):
    grams= 0.
    for cloth in cloth_items:
        grams += deepfashion_dict_grams[cloth]  
    A_cov1 = 0
    for cloth in cloth_areas:
        A_cov1 += cloth

    # A_cov1 = *A_cov1
    clo =0.919 + (0.255*0.001*grams) - (0.00874*A_cov0) - (0.00510*A_cov1)
    return (clo)




device= 'cuda'



def intersection_over_union(box1, box2):
    
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1[2]
    box1_y2 = box1[3]
    box2_x1 = box2[0]
    box2_y1 = box2[1]
    box2_x2 = box2[2]
    box2_y2 = box2[3]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(pred_scores,pred_boxes,pred_label,pred_mask, iou_threshold, threshold):
    pred_scores_filter=[]
    pred_boxes_filter=[]
    pred_label_filter=[]
    pred_mask_filter=[]
    for i in range(len(pred_scores)):
      if pred_scores[i]>threshold:
        pred_scores_filter.append(pred_scores[i])
        pred_boxes_filter.append(pred_boxes[i])
        pred_label_filter.append(pred_label[i])
        pred_mask_filter.append(pred_mask[:,:,i])     
    pred_boxes_filter=[list(x) for x in pred_boxes_filter]
    bboxes_after_nms = []
    copied_bboxes = copy.deepcopy(pred_boxes_filter)
    while copied_bboxes:
        chosen_box = copied_bboxes.pop(0)
        bb=[]
        for b in range(len(copied_bboxes)):
          Iou=intersection_over_union(chosen_box, copied_bboxes[b])
          if Iou < iou_threshold:
            bb.append(copied_bboxes[b])      
        copied_bboxes=bb
        bboxes_after_nms.append(chosen_box)
    pred_scores_nms=[]
    pred_boxes_nms=[]
    pred_label_nms=[]
    pred_mask_nms=[]  
    for box in bboxes_after_nms:
      idx= pred_boxes_filter.index(box)
      pred_boxes_nms.append(box)
      pred_scores_nms.append(pred_scores_filter[idx])
      pred_label_nms.append(pred_label_filter[idx])
      pred_mask_nms.append(pred_mask_filter[idx])
    pred_mask_filter=np.array(pred_mask_nms)
    pred_boxes_filter=np.array(pred_boxes_nms)
    pred_scores_filter=np.array(pred_scores_nms)
    pred_label_filter=np.array(pred_label_nms)    
    return pred_mask_filter,pred_boxes_filter,pred_scores_filter,pred_label_filter


def area_covered_info(image,person_info,pred_outfit_mask,pred_outfit_boxes,pred_outfit_label, masked_image):
  class_names=list(utils_deepfashion.deepfashion_dict.values())
  masked_image= cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
  match_info={}
  auto_show = False
  _, ax = plt.subplots(1, figsize=(14,14))
  auto_show = True
  height, width = masked_image.shape[:2]
  ax.set_ylim(height + 10, -10)
  ax.set_xlim(-10, width + 10)
  ax.axis('off')
  # ax.set_title(title)
  for p in person_info: 
    person_box=tuple(person_info[p]['box'].cpu().detach().numpy())
    match_info[person_box]=[]
    cloth_info =pred_outfit_label
    cloth_label= []
    #draw_bounding_box(image, person_box)
    for b in range(len(pred_outfit_boxes)):
      Iou=intersection_over_union(pred_outfit_boxes[b],person_box)
      #draw_bounding_box(image, pred_outfit_boxes[b])
      #print("Iou for {} is {}.".format(class_names[pred_outfit_label[b]-1],Iou))
      if Iou>0.1:
        label = cloth_info[b]
        label = utils_deepfashion.deepfashion_dict[label]
        cloth_label.append(label)
        match_info[person_box].append(b)
    #print("match_info",match_info) 
    person_mask = person_info[p]['mask'].reshape((224, 224, 1)) 
    covered_area=0 
    cloth_area=[]
    if len(match_info[person_box])>0:  
      for b in match_info[person_box]:
        outfit_mask=pred_outfit_mask[b].reshape((224, 224, 1))
        area=evaluation_metrics.compute_overlaps_masks(person_mask,outfit_mask)[0,0]
        cloth_area.append(area*100)
        covered_area+=area*100
      print("Covered area for {} is {}.".format(p,covered_area))
    A_0 = 100-covered_area
    clo = get_cloth(cloth_items= cloth_label,cloth_areas= cloth_area, A_cov0=A_0)
    print("Clo for {} is {}.".format(p,clo))
    x1, y1, x2, y2 = person_info[p]['box'].cpu().detach()
    pa = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor='red', facecolor='none')
    ax.add_patch(pa)
    # masked_image= visualize.draw_bounding_box(masked_image,person_info[p]['box'])
    caption= p
    
    font_size= 12
    anchor=3
    ax.text(x1, y1+anchor, caption,
                color='w', size=font_size, backgroundcolor="none")
    
    caption2=  f'$A_{0}$ ={A_0.round(2)}'
    ax.text(x1, y1+anchor+4, caption2,
                color='w', size=font_size, backgroundcolor="none")
    

    caption3 = f'$A_{1}$={sum(cloth_area).round(2)}(total)'
    ax.text(x1, y1+anchor+8, caption3,
                color='w', size=font_size, backgroundcolor="none")
    
    
    caption4= f'Clo = {clo.round(2)}' 
    ax.text(x1, y1+anchor+12, caption4,
                color='w', size=font_size, backgroundcolor="none")
    #print (cloth_area)
    plt.imshow(masked_image)
    # plt.show()
  return masked_image,  


def cloth_visualization(path_image, imageSize, model):
        image = cv2.imread(path_image)
        image=cv2.resize(image, (imageSize))
        prediction =  utils_deepfashion.inference(model, path_image, imageSize= imageSize)[0]
        pred_boxes=prediction['boxes'].cpu().numpy()
        pred_label=prediction['labels'].cpu().numpy()
        pred_scores=prediction['scores'].cpu().numpy()
        # print (prediction)
        mask_tensor = prediction['masks']
        class_names= utils_deepfashion.class_names
        mask_tensor=torch.squeeze(mask_tensor,dim=1)
        mask_tensor.shape
        reshaped_mask = mask_tensor.permute(1,2,0)
        pred_mask=reshaped_mask.cpu().numpy()
        # person_info=person_detection(path_image)
        person_info = person_segmentation.person_sgg.persons_sgg_im(path_image, device)
        pred_mask_filter,pred_boxes_filter,pred_scores_filter,pred_label_filter=non_max_suppression(pred_scores,pred_boxes,pred_label,pred_mask, 0.25, 0.5)
        
        
        masked_image= visualize.display_instances(image, pred_boxes_filter, pred_mask_filter, pred_label_filter, class_names,
                            scores=pred_scores_filter, title="",
                            figsize=(6, 8), ax=None,
                            show_mask=True, show_bbox=True,
                            colors=None, captions=None) 

        masked_image= masked_image.astype('uint8')
        plt.imshow(masked_image)
        plt.show()
        # masked_image= area_covered_info(image,person_info,pred_mask_filter,pred_boxes_filter,pred_label_filter, masked_image)

        return masked_image