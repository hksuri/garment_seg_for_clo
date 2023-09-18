# -*- coding: utf-8 -*-
"""Mask_rcnn_deepFashion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o1FpNzDrjEWUo8KEx7e_N7f7oIGnfWAn
"""


"""Reference: https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3

https://medium.com/analytics-vidhya/a-simple-guide-to-maskrcnn-custom-dataset-implementation-27f7eab381f2
"""

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
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
import utils_deepfashion
import sys

file_name= sys.argv[0]
file_name = file_name.split('/')[-1]
file_name = file_name.split('.')[0]

driver_location= 'cluster'
dict_directory={}

root_project = dict_directory[driver_location]
os.chdir(root_project)

# !pip install wget
# import wget
# URL = 'https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1'
# response = wget.download(URL, "labpics.zip")

# !unzip '/content/drive/MyDrive/CS444_Final_Project/Nidia/labpics.zip'
# root_project= root #'/content/drive/MyDrive/CS444_Final_Project/Nidia'



root_dataset = os.path.join(root_project, 'train')#'/content/drive/MyDrive/deepfashion'
train_root =  root_dataset  #os.path.join(root_dataset,'train' )
PATH_TRAIN_images= os.path.join(train_root,'image')
PATH_TRAIN_annotations= (os.path.join(train_root, 'annos'))


PATH_EVALUATION_images= os.path.join(root_project,'validation/image') ## FOR NOW
PATH_EVALUATION_annotations= (os.path.join(root_project, 'validation/annos'))  ## FOR NOW
N_images_evaluation= len(os.listdir(PATH_EVALUATION_images))
sample_evaluation= 300


n_epochs=100
imageSize= (254,254)
num_classes=13
batch_size= 8
list_images= os.listdir(PATH_TRAIN_images)
list_annotations= os.listdir(PATH_TRAIN_annotations)
N_images= len(list_images)



data_loader_params= {}
data_loader_params['list_images']= list_images
data_loader_params['PATH_TRAIN_images']= PATH_TRAIN_images
data_loader_params['PATH_TRAIN_annotations']= PATH_TRAIN_annotations
data_loader_params['imageSize']= imageSize



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print (device)   # train on the GPU or on the CPU, if a GPU is not available
model_directory = os.path.join(root_project, file_name)   ## change the name!!!

try:
   os.mkdir(model_directory)
   print('myDirectory created')
except FileExistsError:
   print('myDirectory already exists')


LOSS_PATH= os.path.join(model_directory,'loss.csv')
if exists(LOSS_PATH):
  summary_loss= pd.read_csv(LOSS_PATH)
  start_epoch= len(summary_loss)

else:
  summary_loss= pd.DataFrame()
  start_epoch= 0

# Load the pretrained SqueezeNet1_1 backbone.
backbone = torchvision.models.squeezenet1_1(pretrained=True).features
# We need the output channels of the last convolutional layers from
# the features for the Faster RCNN model.
# It is 512 for SqueezeNet1_0.
backbone.out_channels = 512
# Generate anchors using the RPN. Here, we are using 5x3 anchors.
# Meaning, anchors with 5 different sizes and 3 different aspect 
# ratios.
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
# Feature maps to perform RoI cropping.
# If backbone returns a Tensor, `featmap_names` is expected to
# be [0]. We can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                     output_size=14,
                                                     sampling_ratio=2)
model = MaskRCNN(backbone, num_classes=14,
             rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
              mask_roi_pool=mask_roi_pooler, trainable_backbone_layers=0,mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])
model.to(device)# move model to the right devic

last_checkpoint_path = os.path.join(model_directory, f"last_checkpoint_{file_name}.torch")
if exists(last_checkpoint_path):
    model.load_state_dict(torch.load(last_checkpoint_path))

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
model.train()
best_loss= float("inf")
best_iou = float('-inf')
best_map = float("-inf")
for epoch in range(start_epoch, n_epochs):
        epoch_loss= []
        epoch_summary=pd.DataFrame()
        permut= np.random.permutation(N_images)
        model.train()
        for i in range(0,int(N_images/10),batch_size): ## REMOVE THE /2. I am doing this because it is taking for ever to debug.
              to_load= permut[i:i+batch_size]
              images, targets = utils_deepfashion.load_images(to_load, **data_loader_params)
              images = list(image.to(device) for image in images)
              targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
              optimizer.zero_grad()
              loss_dict = model(images, targets)
              
              losses = sum(loss for loss in loss_dict.values())
              losses.backward()
              optimizer.step()
              
              df= utils_deepfashion.dict_loss_dataFrame(loss_dict)
              df['epoch']= epoch
              epoch_summary= pd.concat([epoch_summary, df])
              epoch_loss.append(losses.item())

        # print(epoch,'loss:', losses.item())
        mean_loss= np.mean(epoch_loss)
        print(epoch,'loss:', mean_loss)
        if mean_loss < best_loss:
          best_loss= mean_loss
          torch.save(model.state_dict(), os.path.join(model_directory, f"best_checkpoint_{file_name}.torch"))
  
        torch.save(model.state_dict(), os.path.join(model_directory, f"last_checkpoint_{file_name}.torch"))
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(model_directory, f'{epoch}_ checkpoint_{file_name}.torch'))
        

        map_ = 'None'
        iou_ = 'None'
        if epoch%5==0:
            batch_= np.random.permutation(N_images_evaluation)
            map_= []
            iou_=[]
            for i in range(0,4500,sample_evaluation):
                idxs = batch_[i:i+sample_evaluation]
                map_i,iou_i = utils_deepfashion.evaluate(model, 
                                                PATH_evaluation_annotation=PATH_EVALUATION_annotations, 
                                                PATH_evaluation_images=PATH_EVALUATION_images, imageSize=imageSize,
                                                device='cuda', idxs=idxs )
                
                map_.append(np.mean(map_i))
                iou_.append(np.mean(iou_i))

            map_= np.mean(map_)
            iou_ = np.mean(iou_)

            if map_>best_map:
                best_map=map_
        
            if iou_ >best_iou:
                best_iou = iou_ 
                torch.save(model.state_dict(), os.path.join(model_directory, f"best_iou_checkpoint_{file_name}.torch")) 

        torch.save(model.state_dict(), os.path.join(model_directory, f"last_checkpoint_{file_name}.torch"))
        df =pd.DataFrame(epoch_summary.mean(axis=0)).T
        df['best_map'] = map_
        df['best_iou']= best_iou
        df['mean_loss']= mean_loss
        df['map_val'] = map_
        df['iou_val']= iou_ 
        df['file_name'] = file_name
        

        summary_loss= pd.concat([summary_loss, df])
        summary_loss.to_csv(LOSS_PATH)


