from torch.utils.data import DataLoader
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from geomatch.dataset.university import  get_transforms_train_geomatch as get_transforms_train_geomatch_u
from geomatch.model import TimmModel_Geomatch_5
from geomatch.transforms import get_transforms_val
import cv2
import numpy as np
from torch.utils.data import Dataset
from geomatch.transforms import get_transforms_train_geomatch, get_transforms_val
from geomatch.dataset.vigor import VigorDatasetEval_All
from PIL import Image
import heapq
from torch.utils.data import DataLoader


gpu_ids: tuple = (0,1)
config_img_size = 384
config_model = 'convnext_large.fb_in22k_ft_in1k_384'
checkpoint_start = "/mnt/hdd/cky/checkpoint_univisity/convnext_large.fb_in22k_ft_in1k_384/161257/weights_e3_0.0841.pth"
output_dir = '/mnt/hdd/cky/CKimage/'

batch_size_eval: int = 200
config_img_size = 384
model = TimmModel_Geomatch_5(config_model,
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
                                          max_samples=10
                                          )
dataset_size = len(query_dataset_test_v)
print("使用的vigor数据集的大小：",dataset_size)
    
query_dataloader_test_v = DataLoader(query_dataset_test_v,
                                       batch_size = 96,
                                       num_workers = 32,
                                       shuffle=False,
                                       pin_memory=True)
print("使用的vigor数据集的大小：",len(query_dataloader_test_v))


val_transforms,_, _, _, _ = get_transforms_train_geomatch_u(
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



image_path = "/mnt/hdd/cky/dataset/University-Release/train/street/0839/1.jpg"
image = Image.open(image_path).convert('RGB')  

single_feature = get_single_image_feature(model, image =image , transform = val_transforms, device='cuda')

print(f"Single image feature vector shape: {single_feature.shape}")
print(single_feature)



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
def process_query_dataset(model, dataloader, single_feature, device, topk=5, output_base_dir="matched_pairs"):

    model.eval()

    os.makedirs(output_base_dir, exist_ok=True)

    street_dir = os.path.join(output_base_dir, "street")
    satellite_dir = os.path.join(output_base_dir, "satellite")
    os.makedirs(street_dir, exist_ok=True)
    os.makedirs(satellite_dir, exist_ok=True)
    
    dataset = dataloader.dataset
    best_matches = []  # (similarity, img_idx, angle)
    
    for img_idx in tqdm(range(len(dataset)), desc="查找匹配"):
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
                features = model(processed.unsqueeze(0).to(device))
                similarity = torch.nn.functional.cosine_similarity(
                    features, single_feature.unsqueeze(0)
                ).item()
            
            if len(best_matches) < topk:
                heapq.heappush(best_matches, (similarity, img_idx, angle))
            elif similarity > best_matches[0][0]:
                heapq.heapreplace(best_matches, (similarity, img_idx, angle))
    

    best_matches = sorted(best_matches, key=lambda x: x[0], reverse=True)
    

    saved_satellites = {}
    

    for rank, (sim, img_idx, angle) in enumerate(best_matches):

        ground_path = dataloader.dataset.images[img_idx]
        raw_ground_img = cv2.imread(ground_path)
        if raw_ground_img is None:
            continue
        raw_ground_img = cv2.cvtColor(raw_ground_img, cv2.COLOR_BGR2RGB)
        

        sat_idx = dataloader.dataset.df_ground.iloc[img_idx]['sat']
        sat_path = dataloader.dataset.idx2sat_path[sat_idx]
        

        sat_filename = os.path.splitext(os.path.basename(sat_path))[0]

        ground_match_dir = os.path.join(street_dir, sat_filename)
        os.makedirs(ground_match_dir, exist_ok=True)

        sat_match_dir = os.path.join(satellite_dir, sat_filename)
        if sat_filename not in saved_satellites:
            os.makedirs(sat_match_dir, exist_ok=True)
            
            sat_img = cv2.imread(sat_path)
            if sat_img is not None:
                sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
                sat_save_path = os.path.join(sat_match_dir, "satellite.jpg")
                cv2.imwrite(sat_save_path, sat_img)
                saved_satellites[sat_filename] = True
            else:
                continue  

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
    
    print(f"\n保存了 {len(best_matches)} 个匹配结果:")
    for rank, (sim, img_idx, angle) in enumerate(best_matches):
        sat_filename = os.path.splitext(os.path.basename(dataloader.dataset.idx2sat_path[dataloader.dataset.df_ground.iloc[img_idx]['sat']]))[0]
        print(f"匹配 {rank+1}: 相似度 = {sim:.4f}, 卫星图 = {sat_filename}, 角度 = {angle}°")
    
    return best_matches

output_dir = "/mnt/hdd/cky/matched_pair"

top_matches = process_query_dataset(
    model=model,
    dataloader=query_dataloader_test_v,
    single_feature=single_feature,
    device="cuda",
    topk=50,  
    output_base_dir=output_dir
)

