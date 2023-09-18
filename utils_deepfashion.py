
import os
import torch
import cv2
import json
import numpy as np
from matplotlib.path import Path
from pandas.core.sorting import defaultdict
import sys
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import pandas as pd
import random

deepfashion_dict = {}
# deepfashion_dict["short sleeve top"]= 1
# deepfashion_dict["long sleeve top"] = 2
# deepfashion_dict["short sleeve outwear"] = 3 
# deepfashion_dict["long sleeve outwear"] = 4
# deepfashion_dict["vest"] = 5
# deepfashion_dict["sling"] = 6
# deepfashion_dict["shorts"] = 7
# deepfashion_dict["trousers"] = 8 
# deepfashion_dict["skirt"] = 9
# deepfashion_dict["short sleeve dress"] = 10
# deepfashion_dict["long sleeve dress"] = 11
# deepfashion_dict["vest dress"] = 12
# deepfashion_dict["sling dress"]= 13

deepfashion_dict[1]= "short sleeve top"
deepfashion_dict[2] = "long sleeve top"
deepfashion_dict[3] = "short sleeve outwear"
deepfashion_dict[4] = "long sleeve outwear"
deepfashion_dict[5] = "vest"
deepfashion_dict[6] = "sling"
deepfashion_dict[7] = "shorts"
deepfashion_dict[8] ="trousers"
deepfashion_dict[9] = "skirt"
deepfashion_dict[10] = "short sleeve dress"
deepfashion_dict[11] = "long sleeve dress"
deepfashion_dict[12] ="vest dress"
deepfashion_dict[13]="sling dress"

class_names=list(deepfashion_dict.values())
def test():
    print ('ho')
def dict_loss_dataFrame(loss_dict):
    for k in loss_dict:
        loss_dict[k] = loss_dict[k].cpu().detach().numpy()
    loss_dict= pd.DataFrame.from_dict(loss_dict, orient='index').T
    return (loss_dict)


def scaled_box(box, image,imageSize, to_plot=False):
  seq = iaa.Sequential([
    iaa.Resize({"height": imageSize[0], "width": imageSize[1]})])
  bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], x2=box[2], y1=box[1], y2=box[3])], shape= image.shape)
  image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
  scaled_box = []
  bbs_aug= bbs_aug[0]
  if to_plot==True:
    scaled_box.append(int(bbs_aug[0][0]))
    scaled_box.append(int(bbs_aug[0][1]))
    scaled_box.append(int(bbs_aug[1][0]))
    scaled_box.append(int(bbs_aug[1][1]))
  else:
    scaled_box.append(bbs_aug[0][0])
    scaled_box.append(bbs_aug[0][1])
    scaled_box.append(bbs_aug[1][0])
    scaled_box.append(bbs_aug[1][1])

  return (scaled_box)

def dict_loss_dataFrame(loss_dict):
    for k in loss_dict:
        loss_dict[k] = loss_dict[k].cpu().detach().numpy()
    loss_dict= pd.DataFrame.from_dict(loss_dict, orient='index').T
    return (loss_dict)

def load_images(to_load, list_images,PATH_TRAIN_images, PATH_TRAIN_annotations, imageSize):
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    # perm = np.random.permutation(N_images)
    for idx in to_load:
        # idx = perm[idx]
        id_image= list_images[idx].split('.jpg')[0]
        annotation= f'{id_image}.json'
        f = open(os.path.join(PATH_TRAIN_annotations,annotation))
        data = json.load(f)
        f.close()

        image = cv2.imread(os.path.join(PATH_TRAIN_images, list_images[idx]))
        keys_= data.keys()
        items= [i for i in keys_ if 'item' in i]

        num_objs= len(items)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        labels= torch.zeros((num_objs), dtype=torch.int64)
        masks=[]
        for obj_ in range(num_objs):
              cloth= items[obj_]
              data_item = data[cloth]
              box = scaled_box(data_item['bounding_box'], image, imageSize)
              boxes[obj_] = torch.tensor(box)
              coord= data_item['segmentation'][0]
              mask = create_mask(coord, image.shape)
              mask =cv2.resize(mask,imageSize,cv2.INTER_NEAREST)

              masks.append(mask)
              category= data_item['category_id']
              labels[obj_] = category
            
        labels= torch.as_tensor(labels,dtype=torch.int64 )
        image = cv2.resize(image, imageSize, cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255


        image = torch.as_tensor(image, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  labels # there is only one class
        data["masks"] = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)

        batch_Imgs.append(image)
        batch_Data.append(data)  #

    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


def create_mask(coord_, shape_img):
  points= []
  for i in range(0, len(coord_),2):
      points.append([coord_[i+1], coord_[i]])

  poly_path=Path(points)
  height=shape_img[0]
  width=shape_img[1]
  x, y = np.mgrid[:height, :width]
  coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

  mask = poly_path.contains_points(coors)
  mask = mask.reshape(height, width)

  X= np.zeros(mask.shape)
  X[mask]=1
  return (X.astype('uint8'))


def evaluate(model, PATH_evaluation_annotation, PATH_evaluation_images, imageSize,device='cuda',samples=1000, idxs = None):

    target_list, pred_list= evaluation(model, PATH_evaluation_annotation, 
                    PATH_evaluation_images, imageSize,device=device, samples=samples, idxs=idxs)

    targets = defaultdict(list)
    targets_masks = defaultdict(list)

    preds = defaultdict(list)
    preds_masks= defaultdict(list)
    image_list = []  # image path list

    for data in target_list:
        image_id = data['image_id']
        labels= data['labels']
        boxes= data['boxes']
        masks= data['masks']
        
        for obj_class, box_, mask_  in zip(labels, boxes, masks):
          class_name= deepfashion_dict[obj_class.item()]
          targets[(image_id, class_name)].append(box_)
          targets_masks[(image_id, class_name)].append(mask_.numpy())
        
    print("---Evaluate model on test samples---")
    sys.stdout.flush()
    for image_dict in pred_list: 
        image_id = image_dict['image_id']
        for box, prob, class_name, mask_ in zip(image_dict['boxes'], image_dict['scores'], image_dict['labels'],image_dict['masks']):
            x1,y1,x2,y2= box.cpu()
            class_name= deepfashion_dict[class_name.item()]
            prob= prob.cpu()
            preds[class_name].append([image_id, prob.item(), x1.item(), y1.item(), x2.item(), y2.item()])
            preds_masks[class_name].append([image_id, mask_.cpu().numpy()[0,:,:]])
    aps = voc_eval(preds, targets, preds_masks,targets_masks, VOC_CLASSES=deepfashion_dict)
    return aps

def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def voc_eval(
    preds, target,preds_masks,target_masks, VOC_CLASSES=deepfashion_dict, threshold=0.5, use_07_metric=False
):
    """
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    """
    aps = []
    iou_masks=[]
    for i, class_ in enumerate(VOC_CLASSES.values()):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        mask= preds_masks[class_]
        if len(pred) == 0:  # No predictions made for this class
            ap = 0.0
            iou=0
            print(
                "---class {} ap {}--- iou {} (no predictions for this class) ".format(
                    class_, ap, iou
                )
            )
            aps += [ap]
            iou_masks += [iou]
            continue
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        pred_m= np.array([x[1] for x in mask])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        pred_m = pred_m[sorted_ind,:]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.0
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        set_masks_target=[]
        set_masks_pred=[]
        for d, image_id in enumerate(image_ids):
            bb = BB[d]
            mm= pred_m[d]
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]
                MMGT= target_masks[(image_id, class_)]
                # if len(BBGT)>1:
                #   continue 
                for bbgt, mmgt in zip(BBGT, MMGT):
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    union = (
                        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                        + (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                        - inters
                    )
                    if union == 0:
                        print(bb, bbgt)

                    # print (mm.shape, mmgt.shape, overlap_mask)
                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        set_masks_target.append(mmgt)
                        set_masks_pred.append(mm)
                        print (len(BBGT), bbgt.shape, bbgt, len(MMGT), BBGT)
                        # BBGT.remove(bbgt)  # bbox has already been used
                        BBGT= [tensor for tensor in BBGT if tensor is not bbgt]
                        if len(BBGT) == 0:
                            del target[
                                (image_id, class_)
                            ]  # delete things that don't have bbox
                        break

                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        fp_masks =  npos - (len(set_masks_pred)) ## masks not detected
        if len(set_masks_target) !=0:
              set_masks_target= np.asarray(set_masks_target)
              set_masks_pred= np.asarray(set_masks_pred)
              set_masks_target = set_masks_target.reshape(set_masks_target.shape[1], set_masks_target.shape[2], set_masks_target.shape[0])
              set_masks_pred = set_masks_pred.reshape(set_masks_pred.shape[1], set_masks_pred.shape[2], set_masks_pred.shape[0])
              iou = compute_overlaps_masks(set_masks_target, set_masks_pred)
              iou = np.trace(iou)/(iou.shape[1]+fp_masks)
        else:
              iou = 0
        
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print("---class {} ap {}---  iou {}".format(class_, ap, iou))
        
        if np.sum(tp) ==0:
            ap=0
        aps += [ap]
        iou_masks += [iou]
    print("---map {}---".format(np.mean(aps)), np.mean(iou_masks))
    return aps, iou_masks

def evaluation(model, PATH_evaluation_annotation, PATH_evaluation_images, imageSize,device='cuda',samples=1000, idxs=None):
        # evaluation_directory=  '/content/drive/MyDrive/deepfashion/'
        # PATH_evaluation_images= os.path.join(evaluation_directory, 'train')
        # PATH_evaluation_annotation = os.path.join(evaluation_directory, 'annos_train_mini')

        list_directory= os.listdir(PATH_evaluation_images)
        if idxs is not None:
            list_directory = np.asarray(list_directory)
            list_directory = list_directory[idxs]
        
        # list_directory= random.sample(list_directory, samples)
        predictions=[]
        targets=[]
        model.eval()

        for img in list_directory:
            image_id=   img.split('.jpg')[0]
            imgPath =  os.path.join(PATH_evaluation_images,img)
            images=cv2.imread(imgPath)
            image_ref = images.copy()

            image_shape= images.shape
            images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)/255

            images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
            images=images.swapaxes(1, 3).swapaxes(2, 3)
            images= images.to(device)
            images = list(image.to(device) for image in images)
            with torch.no_grad():
                pred = model(images)
                

            del images
            pred=pred[0]
            pred['image_id']= image_id
            predictions.append(pred)

            annotation= f'{image_id}.json'
            f = open(os.path.join(PATH_evaluation_annotation,annotation))
            data = json.load(f)
            f.close()

            keys_= data.keys()
            items= [i for i in keys_ if 'item' in i]

            num_objs= len(items)
            boxes = torch.zeros([num_objs,4], dtype=torch.float32)
            labels= torch.zeros((num_objs), dtype=torch.int64)
            masks=[]
            for obj_ in range(num_objs):
                  cloth= items[obj_]
                  data_item = data[cloth]
                  box = scaled_box(data_item['bounding_box'], image_ref, imageSize)
                  boxes[obj_] = torch.tensor(box)
                  coord= data_item['segmentation'][0]
                  mask = create_mask(coord, image_shape)
                  mask =cv2.resize(mask,imageSize,cv2.INTER_NEAREST)

                  masks.append(mask)
                  category= data_item['category_id']
                  labels[obj_] = category
            del image_ref      
            labels= torch.as_tensor(labels,dtype=torch.int64 )

            data = {}
            data["boxes"] =  boxes
            data["labels"] =  labels # there is only one class
            data["masks"] = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)
            data['image_id']= image_id

            targets.append(data)

        return targets,predictions

# def draw_bounding_box(img, box):
    
#     x_min, y_min = box[0],box[1]
#     x_max, y_max = box[2],box[3]
    
#     cv2.rectangle(img, (x_min,y_min), (x_max,y_max), color=(0,255,0), thickness=2)
#     cv2_imshow(image)


def inference(model, path_image, imageSize,device='cuda'):
        # evaluation_directory=  '/content/drive/MyDrive/deepfashion/'
        # PATH_evaluation_images= os.path.join(evaluation_directory, 'train')
        # PATH_evaluation_annotation = os.path.join(evaluation_directory, 'annos_train_mini')

        # list_directory= random.sample(list_directory, samples)
        model.eval()
        predictions=[]
        targets=[]
        images=cv2.imread(path_image)
        image_shape= images.shape
        images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)/255

        images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
        images=images.swapaxes(1, 3).swapaxes(2, 3)
        images= images.to(device)
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            pred = model(images)        

        del images
        pred=pred[0]
        pred['image_id']= path_image
        predictions.append(pred)

        return predictions
def load_images(to_load, list_images,PATH_TRAIN_images, PATH_TRAIN_annotations, imageSize):
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    # perm = np.random.permutation(N_images)
    for idx in to_load:
        # idx = perm[idx]
        id_image= list_images[idx].split('.jpg')[0]
        annotation= f'{id_image}.json'
        f = open(os.path.join(PATH_TRAIN_annotations,annotation))
        data = json.load(f)
        f.close()

        image = cv2.imread(os.path.join(PATH_TRAIN_images, list_images[idx]))
        keys_= data.keys()
        items= [i for i in keys_ if 'item' in i]

        num_objs= len(items)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        labels= torch.zeros((num_objs), dtype=torch.int64)
        masks=[]
        for obj_ in range(num_objs):
              cloth= items[obj_]
              data_item = data[cloth]
              box = scaled_box(data_item['bounding_box'], image, imageSize)
              boxes[obj_] = torch.tensor(box)
              coord= data_item['segmentation'][0]
              mask = create_mask(coord, image.shape)
              mask =cv2.resize(mask,imageSize,cv2.INTER_NEAREST)

              masks.append(mask)
              category= data_item['category_id']
              labels[obj_] = category
            
        labels= torch.as_tensor(labels,dtype=torch.int64 )
        image = cv2.resize(image, imageSize, cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255


        image = torch.as_tensor(image, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  labels # there is only one class
        data["masks"] = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)

        batch_Imgs.append(image)
        batch_Data.append(data)  #

    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data

