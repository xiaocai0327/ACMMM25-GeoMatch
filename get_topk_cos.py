from torch.utils.data import DataLoader
import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from congeo.dataset.university import  get_transforms_train_congeo as get_transforms_train_congeo_u
from congeo.utils import setup_system, Logger
from congeo.evaluate.university import evaluate
# from congeo.evaluate.rerank2 import evaluate
from congeo.model import TimmModel
from congeo.model import TimmModel_ConGeo,TimmModel_ConGeo_5
from congeo.transforms import get_transforms_val
import os
import glob
import random
import copy
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


gpu_ids: tuple = (0,1)
config_img_size = 384
config_model = 'convnext_large.fb_in22k_ft_in1k_384'
checkpoint_start = "/mnt/hdd/cky/checkpoint_univisity/convnext_large.fb_in22k_ft_in1k_384/161257/weights_e3_0.0841.pth"
output_dir = '/mnt/hdd/cky/CKimage/'

batch_size_eval: int = 200
config_img_size = 384
model = TimmModel_ConGeo_5(config_model,
                          pretrained=False,
                          img_size=config_img_size)
data_config = model.get_config()
print(data_config)
mean = data_config["mean"]
std = data_config["std"]
img_size = (config_img_size, config_img_size)
model_state_dict = torch.load(checkpoint_start, map_location="cpu") 
model.load_state_dict(model_state_dict, strict=False) 
model = model.to("cuda")
model = torch.nn.DataParallel(model, device_ids=gpu_ids)  



from congeo.transforms import get_transforms_train_congeo, get_transforms_val
from congeo.dataset.vigor import VigorDatasetEval,VigorDatasetEval_All, VigorDatasetTrainConGeo,VigorDatasetTrainConGeo_All
data_folder_vigor = "/mnt/hdd/lx/VIGOR/"
image_size_sat = (384, 384)
new_width = 384*2    
new_hight = int((1024 / 2048) * new_width)
img_size_ground = (new_hight, new_width)
sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   ground_cutting=0)
query_dataset_test_v = VigorDatasetEval_All(data_folder=data_folder_vigor ,
                                          split="test",
                                          img_type="query",
                                          same_area=True,      
                                          transforms=ground_transforms_val,
                                          max_samples=4000
                                          )
dataset_size = len(query_dataset_test_v)
print("使用的vigor数据集的大小：",dataset_size)
    
query_dataloader_test_v = DataLoader(query_dataset_test_v,
                                       batch_size = 96,
                                       num_workers = 32,
                                       shuffle=False,
                                       pin_memory=True)
print("使用的vigor数据集的大小：",len(query_dataloader_test_v))


val_transforms,_, _, _, _ = get_transforms_train_congeo_u(
        img_size,
        img_size, 
        mean=mean, 
        std=std)

def get_single_image_feature(model, image, transform, device='cuda'):

    model.eval()
    
    with torch.no_grad():
        image = np.array(image) 
        transformed = transform(image=image) 
        image = transformed['image']  
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)  
        image = image.unsqueeze(0) 
        image = image.to(device)  

        feature = model(image)  

    
    return feature.squeeze(0)  

from PIL import Image

image_path = "/mnt/hdd/cky/dataset/University-Release/train/street/0839/1.jpg"
image = Image.open(image_path).convert('RGB')  

single_feature = get_single_image_feature(model, image =image , transform = val_transforms, device='cuda')

print(f"Single image feature vector shape: {single_feature.shape}")
print(single_feature)

import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import heapq
from torch.utils.data import DataLoader

def crop_panorama(img, fov=90, stride=45):
    h, w = img.shape[:2]
    crop_width = int(w * fov / 360) 
    crop_imgs = []
    angles = []

    for start_x in range(0, w, stride):
        end_x = start_x + crop_width
        angle = int(360 * start_x / w)  
        
        if end_x <= w:
            crop = img[:, start_x:end_x]
        else:
            part1 = img[:, start_x:w]
            part2 = img[:, 0:end_x - w]
            crop = np.concatenate((part1, part2), axis=1)
        
        crop_imgs.append(crop)
        angles.append(angle)
    
    return crop_imgs, angles
def process_query_dataset(model, dataloader, single_feature, device, topk=5, output_dir="matched_crops"):
    """
    修复版本：从原始图像直接裁剪保存
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    dataset = dataloader.dataset
    best_matches = [] 
    target_feature = single_feature.to(device).detach()
    for img_idx in tqdm(range(len(dataset)), desc="查找匹配"):
        img_path = dataset.images[img_idx]

        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        crops, angles = crop_panorama(raw_img)

        for crop, angle in zip(crops, angles):
            processed = dataset.transforms(image=crop)['image'] if dataset.transforms else crop
            if not isinstance(processed, torch.Tensor):
                processed = torch.tensor(processed).permute(2, 0, 1).float()
            with torch.no_grad():
                features = model(processed.unsqueeze(0).to(device))
                similarity = torch.nn.functional.cosine_similarity(
                    features, target_feature.unsqueeze(0)
                ).item()

            if len(best_matches) < topk:
                heapq.heappush(best_matches, (similarity, img_idx, angle))
            elif similarity > best_matches[0][0]:
                heapq.heapreplace(best_matches, (similarity, img_idx, angle))
    best_matches = sorted(best_matches, key=lambda x: x[0], reverse=True)
    for idx, (sim, img_idx, angle) in enumerate(best_matches):
        img_path = dataset.images[img_idx]
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        h, w, _ = raw_img.shape
        crop_width = int(w * 90 / 360)
        start_x = int(w * angle / 360)
        end_x = start_x + crop_width
        
        if end_x <= w:
            crop_img = raw_img[:, start_x:end_x]
        else:
            part1 = raw_img[:, start_x:w]
            part2 = raw_img[:, 0:end_x-w]
            crop_img = np.concatenate((part1, part2), axis=1)

        save_path = os.path.join(output_dir, f"match_{idx+1}_sim{sim:.4f}_angle{angle}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    
    return best_matches

output_dir = "/mnt/hdd/cky/top50/0839/topk0839/"

top_matches = process_query_dataset(
    model=model,
    dataloader=query_dataloader_test_v,
    single_feature=single_feature,
    device="cuda",
    topk=50,  
    output_dir=output_dir
)

