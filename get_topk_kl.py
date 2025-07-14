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
from congeo.transforms import get_transforms_val
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tem import ConvNextProbabilisticEmbedding

from PIL import Image


img_size = (512,512)
config_model = 'convnext_large.fb_in22k_ft_in1k_384'
checkpoint_start = "/mnt/hdd/cky/checkpoint_univisity/convnext_large.fb_in22k_ft_in1k_384/161257/weights_e3_0.0841.pth"
batch_size_eval: int = 200
config_img_size = 384
model = ConvNextProbabilisticEmbedding()
saved_path = "/mnt/hdd/cky/geolocalization_models/weights_e1_-0.0000.pth"
saved_state = torch.load(saved_path)
model.load_state_dict(saved_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

from congeo.transforms import get_transforms_train_congeo, get_transforms_val
from congeo.dataset.vigor import VigorDatasetEval,VigorDatasetEval_All, VigorDatasetTrainConGeo,VigorDatasetTrainConGeo_All
data_folder_vigor = "/mnt/hdd/lx/VIGOR/"
image_size_sat = (512, 512)
new_width = 512*2    
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

def get_single_image_features(model, image, transform, device='cuda'):
    model.eval()
    with torch.no_grad():
        image = np.array(image)
        transformed = transform(image=image)
        image_tensor = transformed['image']
        
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        

        if isinstance(outputs, tuple) and len(outputs) == 2:
            mu, sigma = outputs
        else:
            feat_dim = outputs.shape[-1] // 2
            mu = outputs[:, :feat_dim]
            sigma = outputs[:, feat_dim:]
        
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)
    
    return mu, sigma

def kl_divergence(mu1, sigma1, mu2, sigma2):
    var1 = torch.square(sigma1) + 1e-9
    var2 = torch.square(sigma2) + 1e-9
    
    term1 = torch.log(var2 / var1)
    term2 = var1 / var2
    term3 = torch.square(mu1 - mu2) / var2
    kl = 0.5 * torch.sum(term1 + term2 + term3 - 1)
    return kl.item()

def process_query_dataset_kl(model, dataloader, single_mu, single_sigma, device='cuda', topk=5, output_base_dir="kl_dataset"):
    model.eval()
    os.makedirs(output_base_dir, exist_ok=True)
    street_dir = os.path.join(output_base_dir, "street")
    satellite_dir = os.path.join(output_base_dir, "satellite")
    os.makedirs(street_dir, exist_ok=True)
    os.makedirs(satellite_dir, exist_ok=True)
    dataset = dataloader.dataset
    best_matches = []  
    for img_idx in tqdm(range(len(dataset)), desc="基于KL散度的匹配"):
        img_path = dataset.images[img_idx]
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        crops, angles = crop_panorama(raw_img)
        for crop, angle in zip(crops, angles):
            processed = dataset.transforms(image=crop)['image'] if dataset.transforms else crop
            if not isinstance(processed, torch.Tensor):
                processed = torch.tensor(processed).permute(2, 0, 1).float()
            with torch.no_grad():
                processed = processed.unsqueeze(0).to(device)
                outputs = model(processed)
                
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    mu_v, sigma_v = outputs
                else:
                    feat_dim = outputs.shape[-1] // 2
                    mu_v = outputs[:, :feat_dim]
                    sigma_v = outputs[:, feat_dim:]
                mu_v = mu_v.squeeze(0)
                sigma_v = sigma_v.squeeze(0)
                kl = kl_divergence(mu_v, sigma_v, single_mu, single_sigma)

            if len(best_matches) < topk:
                heapq.heappush(best_matches, (-kl, kl, img_idx, angle))
            elif kl < -best_matches[0][0]:
                heapq.heapreplace(best_matches, (-kl, kl, img_idx, angle))

    best_matches = sorted(best_matches, key=lambda x: x[1])

    match_results = []
    saved_satellites = {}  
    
    for rank, (neg_kl, kl_score, img_idx, angle) in enumerate(best_matches):
        ground_path = dataloader.dataset.images[img_idx]
        raw_ground_img = cv2.imread(ground_path)
        if raw_ground_img is None:
            continue
        raw_ground_img = cv2.cvtColor(raw_ground_img, cv2.COLOR_BGR2RGB)
        
        sat_idx = dataloader.dataset.df_ground.iloc[img_idx]['sat']
        sat_path = dataloader.dataset.idx2sat_path[sat_idx]
        sat_img = cv2.imread(sat_path)
        if sat_img is None:
            continue
        
        sat_filename = os.path.splitext(os.path.basename(sat_path))[0]
        ground_match_dir = os.path.join(street_dir, sat_filename)
        os.makedirs(ground_match_dir, exist_ok=True)
        sat_match_dir = os.path.join(satellite_dir, sat_filename)
        if sat_filename not in saved_satellites:
            os.makedirs(sat_match_dir, exist_ok=True)
            
            sat_save_path = os.path.join(sat_match_dir, "satellite.jpg")
            cv2.imwrite(sat_save_path, sat_img)
            saved_satellites[sat_filename] = True
        
        h, w, _ = raw_ground_img.shape
        crop_width = int(w * 90 / 360)
        start_x = int(w * angle / 360)
        end_x = start_x + crop_width
        
        if end_x <= w:
            crop_img = raw_ground_img[:, start_x:end_x]
        else:
            part1 = raw_ground_img[:, start_x:w]
            part2 = raw_ground_img[:, 0:end_x - w]
            crop_img = np.concatenate((part1, part2), axis=1)
        ground_save_path = os.path.join(ground_match_dir, f"ground_{rank+1}_angle{angle}.jpg")
        cv2.imwrite(ground_save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        match_results.append({
            "rank": rank + 1,
            "kl_score": kl_score,
            "satellite": sat_filename,
            "angle": angle,
            "ground_path": ground_path,
            "sat_path": sat_path,
            "ground_image": f"ground_{rank+1}_angle{angle}.jpg",
            "satellite_image": "satellite.jpg"
        })
    
    # 创建汇总信息文件
    results_df = pd.DataFrame(match_results)
    results_df.to_csv(os.path.join(output_base_dir, "matching_results.csv"), index=False)

    return best_matches

image_path = "/mnt/hdd/cky/dataset/University-Release/train/street/0842/1.jpg"
image = Image.open(image_path).convert('RGB') 

single_mu, single_sigma = get_single_image_features(
    model, 
    image=image, 
    transform=val_transforms, 
    device='cuda'
)


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

output_dir = "/mnt/hdd/cky/result_kl/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
top_matches_kl = process_query_dataset_kl(
    model=model,
    dataloader=query_dataloader_test_v,
    single_mu=single_mu,
    single_sigma=single_sigma,
    device="cuda",
    topk=50,
    output_base_dir=output_dir
)

print("\nTop KL matches:")
for idx, (neg_kl, kl_score, img_idx, angle) in enumerate(top_matches_kl):
    print(f"Match {idx+1}: KL = {kl_score:.4f}, Image index = {img_idx}, Angle = {angle}°")
