import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import glob
import os
import glob
import pandas as pd
import cv2
import random
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
                
    return data

class U1652DatasetTrain(Dataset):
    
    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()
 

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        
        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        
        self.pairs = []
        
        for idx in self.ids:
            
            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            for g in gallery_imgs:
                self.pairs.append((idx, query_img, "{}/{}".format(gallery_path, g)))
        
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        
        self.samples = copy.deepcopy(self.pairs)
        
    def __getitem__(self, index):
        
        idx, query_img_path, gallery_img_path = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, idx

    def __len__(self):
        return len(self.samples)
    
    
    def shuffle(self, ):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
              
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
             
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    idx, _, _ = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
            
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 512:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  
    
        
        
class U1652DatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
 

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())
                

        self.transforms = transforms
        
        self.given_sample_ids = sample_ids
        
        self.images = []
        self.sample_ids = []
        
        self.mode = mode
        
        
        self.gallery_n = gallery_n
        

        for i, sample_id in enumerate(self.ids):
                
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                    
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                      file))
                
                self.sample_ids.append(sample_id) 
                    
  
            
        
        
    def __getitem__(self, index):
        
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1
        
        return img, label

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)
    
    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                                
                             
                                
    
    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*img_size[0]),
                                                               max_width=int(0.2*img_size[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*img_size[0]),
                                                               min_width=int(0.1*img_size[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(max_holes=25,
                                                                 max_height=int(0.2*img_size[0]),
                                                                 max_width=int(0.2*img_size[0]),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*img_size[0]),
                                                                 min_width=int(0.1*img_size[0]),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms




def get_transforms_train_geomatch(image_size_sat,img_size_ground,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],):
    

    satellite_transforms = A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=10, min_holes=3, max_height=16, max_width=16, p=0.5),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])


    satellite_transforms_con = A.Compose([

        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.7), 
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),  
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.5),  
        A.OneOf([
            A.GridDropout(ratio=0.5, p=1.0),
            A.CoarseDropout(max_holes=15, min_holes=5, max_height=16, max_width=16, p=0.5),
        ], p=0.5),  
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    street_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.5, p=1.0),
            A.CoarseDropout(max_holes=10, min_holes=3, max_height=16, max_width=16, p=0.5),
        ], p=0.3),
         A.Normalize(mean, std),
        ToTensorV2(),
    ])


    street_transforms_con = A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.7),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.7), 
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GridDropout(ratio=0.6, p=1.0),
            A.CoarseDropout(max_holes=15, min_holes=5, max_height=16, max_width=16, p=0.5),
        ], p=0.5),
        A.Rotate(limit=30, p=0.5),  
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([A.Resize(image_size_sat[0],image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                
    return val_transforms,satellite_transforms, satellite_transforms_con, street_transforms, street_transforms_con





class U1652DatasetTrainGeomatch(Dataset):
    def __init__(self, data_folder, transforms_query=None, transforms_drone=None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):

        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        

        ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))
        ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        

        sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))
        

        drone_files = sorted(glob.glob(os.path.join(data_folder,"train", "drone", "*", "*.jpeg")))
        drone_ids = [os.path.basename(os.path.dirname(f)) for f in drone_files]
        

        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")
        if not drone_files:
            print(f"警告: No drone images found in {os.path.join(data_folder, '/train/drone')}, using street images as fallback")
        

        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })
        df_drone = pd.DataFrame({
            "drone": [os.path.basename(f) for f in drone_files],
            "sat": [f"{id}.jpg" for id in drone_ids],
            "path_drone": drone_files,
            "location_id": drone_ids
        })
        

        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)
        df_drone["sat"] = df_drone["sat"].map(sat2idx)
        

        unmapped_ground = df_ground["sat"].isna().sum()
        unmapped_drone = df_drone["sat"].isna().sum()
        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")
        if unmapped_drone > 0:
            print(f"警告: {unmapped_drone} 张无人机图像无法匹配卫星")
        
        self.df_ground = df_ground
        self.df_drone = df_drone
        self.idx2ground_path = df_ground["path_ground"].to_dict()
        self.idx2drone_path = df_drone["path_drone"].to_dict()
        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()
        self.drone_location_indices = df_drone.groupby("location_id")["path_drone"].apply(list).to_dict()
        
        # 过滤有效对
        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query = transforms_query
        self.transforms_drone = transforms_drone
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()
        

        missing_drones = sum(1 for loc in set(self.idx2location.values()) if loc not in self.drone_location_indices)
        print(f"Missing drone locations: {missing_drones}/{len(set(self.idx2location.values()))}")
    
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        print("Initial pair_pool size:", len(pair_pool))
        if len(pair_pool) == 0:
            raise ValueError("No street-satellite pairs available. Check dataset loading.")
        random.shuffle(pair_pool)
        self.samples = pair_pool
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        drone_paths = self.drone_location_indices.get(location_id, [])
        if len(drone_paths) > 0:
            drone_path = random.choice(drone_paths)
            drone_img = cv2.imread(drone_path)
            if drone_img is None:
                raise ValueError(f"Failed to load drone image: {drone_path}")
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        else:
            drone_img = query_img.copy()
            print(f"警告: 地点 {location_id} 无无人机图像，使用街景替代")
            if np.random.random() < self.prob_flip:
                drone_img = cv2.flip(drone_img, 1)
            elif np.random.random() < self.prob_rotate:
                angle = random.choice([90, 180, 270])
                if angle == 90:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_180)
                elif angle == 270:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        query_img = self.transforms_query(image=query_img)['image']
        drone_img = self.transforms_drone(image=drone_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img, drone_img, reference_img1, reference_img2, label
    
    def __len__(self):
        return len(self.samples)




class U1652DatasetTrainGeomatch_5(Dataset):
    def __init__(self, data_folder, transforms_query=None, transforms_drone=None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):

        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        

        ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))

        ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        

        sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))
        

        drone_files = sorted(glob.glob(os.path.join(data_folder,"train", "drone", "*", "*.jpeg")))
        drone_ids = [os.path.basename(os.path.dirname(f)) for f in drone_files]
        

        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")

        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })
        df_drone = pd.DataFrame({
            "drone": [os.path.basename(f) for f in drone_files],
            "sat": [f"{id}.jpg" for id in drone_ids],
            "path_drone": drone_files,
            "location_id": drone_ids
        })
        

        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)
        df_drone["sat"] = df_drone["sat"].map(sat2idx)
        

        unmapped_ground = df_ground["sat"].isna().sum()
        unmapped_drone = df_drone["sat"].isna().sum()
        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")
        if unmapped_drone > 0:
            print(f"警告: {unmapped_drone} 张无人机图像无法匹配卫星")
        
        self.df_ground = df_ground
        self.df_drone = df_drone
        self.idx2ground_path = df_ground["path_ground"].to_dict()
        self.idx2drone_path = df_drone["path_drone"].to_dict()
        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()
        self.drone_location_indices = df_drone.groupby("location_id")["path_drone"].apply(list).to_dict()


        self.location_to_ground_indices = df_ground.groupby("location_id").apply(lambda x: x.index.tolist()).to_dict()        
        

        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query = transforms_query
        self.transforms_drone = transforms_drone
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()
        

        missing_drones = sum(1 for loc in set(self.idx2location.values()) if loc not in self.drone_location_indices)
        print(f"Missing drone locations: {missing_drones}/{len(set(self.idx2location.values()))}")
    

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        '''
        Custom shuffle function for unique class_id and location_id sampling in batch
        '''
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)
        pairs_epoch = set()
        idx_batch = set()
        location_batch = set()  
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()
        while True:
            pbar.update()
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, sat = pair
                location_id = self.idx2location.get(idx, None)
                if (idx not in idx_batch and location_id not in location_batch and 
                    pair not in pairs_epoch and location_id is not None):
                    idx_batch.add(idx)
                    location_batch.add(location_id)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1
                    if break_counter >= 512:
                        break
            else:
                break
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                location_batch = set()
                current_batch = []
        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))
        else:
            print("No samples after shuffle.")

    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        ground_indices = self.location_to_ground_indices[location_id]
        if len(ground_indices) > 1:

            other_indices = [idx for idx in ground_indices if idx != idx_ground]
            additional_idx_ground = random.choice(other_indices)
        else:

            additional_idx_ground = idx_ground
        additional_query_path = self.idx2ground_path[additional_idx_ground]
        additional_query_img = cv2.imread(additional_query_path)
        if additional_query_img is None:
            raise ValueError(f"Failed to load additional street image: {additional_query_path}")
        additional_query_img = cv2.cvtColor(additional_query_img, cv2.COLOR_BGR2RGB)
        additional_query_img = self.transforms_query(image=additional_query_img)['image']


        drone_paths = self.drone_location_indices.get(location_id, [])
        if len(drone_paths) > 0:
            drone_path = random.choice(drone_paths)
            drone_img = cv2.imread(drone_path)
            if drone_img is None:
                raise ValueError(f"Failed to load drone image: {drone_path}")
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        else:
            drone_img = query_img.copy()
            print(f"警告: 地点 {location_id} 无无人机图像，使用街景替代")
            if np.random.random() < self.prob_flip:
                drone_img = cv2.flip(drone_img, 1)
            elif np.random.random() < self.prob_rotate:
                angle = random.choice([90, 180, 270])
                if angle == 90:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_180)
                elif angle == 270:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        query_img = self.transforms_query(image=query_img)['image']
        drone_img = self.transforms_drone(image=drone_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img,drone_img,reference_img1, reference_img2,additional_query_img, label
    
    def __len__(self):
        return len(self.samples)




class U1652DatasetTrainGeomatch_g2g_d(Dataset):
    def __init__(self, data_folder, transforms_query=None, transforms_drone=None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):

        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        

        ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))

        ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        

        sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))
        

        drone_files = sorted(glob.glob(os.path.join(data_folder,"train", "drone", "*", "*.jpeg")))
        drone_ids = [os.path.basename(os.path.dirname(f)) for f in drone_files]
        

        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")
        if not drone_files:
            print(f"警告: No drone images found in {os.path.join(data_folder, '/train/drone')}, using street images as fallback")
        


        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })
        df_drone = pd.DataFrame({
            "drone": [os.path.basename(f) for f in drone_files],
            "sat": [f"{id}.jpg" for id in drone_ids],
            "path_drone": drone_files,
            "location_id": drone_ids
        })

        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)
        df_drone["sat"] = df_drone["sat"].map(sat2idx)
        

        unmapped_ground = df_ground["sat"].isna().sum()
        unmapped_drone = df_drone["sat"].isna().sum()
        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")
        if unmapped_drone > 0:
            print(f"警告: {unmapped_drone} 张无人机图像无法匹配卫星")
        
        self.df_ground = df_ground
        self.df_drone = df_drone
        self.idx2ground_path = df_ground["path_ground"].to_dict()
        self.idx2drone_path = df_drone["path_drone"].to_dict()
        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()
        self.drone_location_indices = df_drone.groupby("location_id")["path_drone"].apply(list).to_dict()

        self.location_to_ground_indices = df_ground.groupby("location_id").apply(lambda x: x.index.tolist()).to_dict()        
        

        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query = transforms_query
        self.transforms_drone = transforms_drone
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()
        

        missing_drones = sum(1 for loc in set(self.idx2location.values()) if loc not in self.drone_location_indices)
        print(f"Missing drone locations: {missing_drones}/{len(set(self.idx2location.values()))}")
    
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        print("Initial pair_pool size:", len(pair_pool))
        if len(pair_pool) == 0:
            raise ValueError("No street-satellite pairs available. Check dataset loading.")
        random.shuffle(pair_pool)
        self.samples = pair_pool
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        ground_indices = self.location_to_ground_indices[location_id]
        if len(ground_indices) > 1:

            other_indices = [idx for idx in ground_indices if idx != idx_ground]
            additional_idx_ground = random.choice(other_indices)
        else:

            additional_idx_ground = idx_ground
        additional_query_path = self.idx2ground_path[additional_idx_ground]
        additional_query_img = cv2.imread(additional_query_path)
        if additional_query_img is None:
            raise ValueError(f"Failed to load additional street image: {additional_query_path}")
        additional_query_img = cv2.cvtColor(additional_query_img, cv2.COLOR_BGR2RGB)
        additional_query_img = self.transforms_query(image=additional_query_img)['image']


        drone_paths = self.drone_location_indices.get(location_id, [])
        if len(drone_paths) > 0:
            drone_path = random.choice(drone_paths)
            drone_img = cv2.imread(drone_path)
            if drone_img is None:
                raise ValueError(f"Failed to load drone image: {drone_path}")
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        else:
            drone_img = query_img.copy()
            print(f"警告: 地点 {location_id} 无无人机图像，使用街景替代")
            if np.random.random() < self.prob_flip:
                drone_img = cv2.flip(drone_img, 1)
            elif np.random.random() < self.prob_rotate:
                angle = random.choice([90, 180, 270])
                if angle == 90:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_180)
                elif angle == 270:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        query_img = self.transforms_query(image=query_img)['image']
        drone_img = self.transforms_drone(image=drone_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img, drone_img, reference_img1, reference_img2,additional_query_img, label
    
    def __len__(self):
        return len(self.samples)





class U1652DatasetTrainGeomatch_5_google(Dataset):
    def __init__(self, data_folder, transforms_query=None, transforms_drone=None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):

        use_google = True

        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        
        if use_google:

            street_files = sorted(glob.glob(os.path.join(data_folder, "train", "street", "*", "*.jpg")))

            google_files = sorted(glob.glob(os.path.join(data_folder, "train", "google_ss", "*", "*.jpg")))

            ground_files = street_files + google_files

            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        else:

            ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))

            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        

        sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))
        

        drone_files = sorted(glob.glob(os.path.join(data_folder,"train", "drone", "*", "*.jpeg")))
        drone_ids = [os.path.basename(os.path.dirname(f)) for f in drone_files]
        

        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")


        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })



        df_drone = pd.DataFrame({
            "drone": [os.path.basename(f) for f in drone_files],
            "sat": [f"{id}.jpg" for id in drone_ids],
            "path_drone": drone_files,
            "location_id": drone_ids
        })
        

        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)
        df_drone["sat"] = df_drone["sat"].map(sat2idx)
        

        unmapped_ground = df_ground["sat"].isna().sum()
        unmapped_drone = df_drone["sat"].isna().sum()
        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")
        if unmapped_drone > 0:
            print(f"警告: {unmapped_drone} 张无人机图像无法匹配卫星")
        
        self.df_ground = df_ground
        self.df_drone = df_drone
        self.idx2ground_path = df_ground["path_ground"].to_dict()
        self.idx2drone_path = df_drone["path_drone"].to_dict()
        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()
        self.drone_location_indices = df_drone.groupby("location_id")["path_drone"].apply(list).to_dict()


        self.location_to_ground_indices = df_ground.groupby("location_id").apply(lambda x: x.index.tolist()).to_dict()        
        

        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query = transforms_query
        self.transforms_drone = transforms_drone
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()
        

        missing_drones = sum(1 for loc in set(self.idx2location.values()) if loc not in self.drone_location_indices)
        print(f"Missing drone locations: {missing_drones}/{len(set(self.idx2location.values()))}")
    

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        '''
        Custom shuffle function for unique class_id and location_id sampling in batch
        '''
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)
        pairs_epoch = set()
        idx_batch = set()
        location_batch = set()  
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()
        while True:
            pbar.update()
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, sat = pair
                location_id = self.idx2location.get(idx, None)
                if (idx not in idx_batch and location_id not in location_batch and 
                    pair not in pairs_epoch and location_id is not None):
                    idx_batch.add(idx)
                    location_batch.add(location_id)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1
                    if break_counter >= 512:
                        break
            else:
                break
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                location_batch = set()
                current_batch = []
        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))
        else:
            print("No samples after shuffle.")
    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        ground_indices = self.location_to_ground_indices[location_id]

        drone_paths = self.drone_location_indices.get(location_id, [])
        if len(drone_paths) > 0:
            drone_path = random.choice(drone_paths)
            drone_img = cv2.imread(drone_path)
            if drone_img is None:
                raise ValueError(f"Failed to load drone image: {drone_path}")
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        else:
            drone_img = query_img.copy()
            print(f"警告: 地点 {location_id} 无无人机图像，使用街景替代")
            if np.random.random() < self.prob_flip:
                drone_img = cv2.flip(drone_img, 1)
            elif np.random.random() < self.prob_rotate:
                angle = random.choice([90, 180, 270])
                if angle == 90:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_180)
                elif angle == 270:
                    drone_img = cv2.rotate(drone_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        additional_query_img = self.transforms_drone(image=query_img)['image']
        query_img = self.transforms_query(image=query_img)['image']
        drone_img = self.transforms_drone(image=drone_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img,drone_img,reference_img1, reference_img2,additional_query_img, label
    
    def __len__(self):
        return len(self.samples)







class U1652DatasetTrainGeomatch_vigor(Dataset):
    def __init__(self, data_folder, use_vigor = True,transforms_query1=None,transforms_query2 = None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):


        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        
        if use_vigor:
            street_files = sorted(glob.glob(os.path.join(data_folder, "train", "street", "*", "*.jpg")))
            vigor_g = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/street/", "*", "*_most_similar_block.jpg")))
            ground_files = street_files + vigor_g
            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        else:

            ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))

            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        
        if use_vigor:

            sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))

            vigor_s = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/satellite/", "*", "*.png")))

            sat_files = sat_files+ vigor_s

        else:

            sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))

        vigor_only = True
        if vigor_only == True:
            ground_files = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/street/", "*", "*_most_similar_block.jpg")))
            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
            sat_files = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/satellite/", "*", "*.png")))


        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")

        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })


        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)

        
        # 检查未匹配
        unmapped_ground = df_ground["sat"].isna().sum()
        # unmapped_drone = df_drone["sat"].isna().sum()
        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")

        
        self.df_ground = df_ground
        # self.df_drone = df_drone
        self.idx2ground_path = df_ground["path_ground"].to_dict()
        # self.idx2drone_path = df_drone["path_drone"].to_dict()
        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()

        self.location_to_ground_indices = df_ground.groupby("location_id").apply(lambda x: x.index.tolist()).to_dict()        
        
        # 过滤有效对
        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2
        # self.transforms_drone = transforms_drone
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()
        

        # missing_drones = sum(1 for loc in set(self.idx2location.values()) if loc not in self.drone_location_indices)
        # print(f"Missing drone locations: {missing_drones}/{len(set(self.idx2location.values()))}")
    
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        print("Initial pair_pool size:", len(pair_pool))
        if len(pair_pool) == 0:
            raise ValueError("No street-satellite pairs available. Check dataset loading.")
        random.shuffle(pair_pool)
        self.samples = pair_pool
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        ground_indices = self.location_to_ground_indices[location_id]
        if len(ground_indices) > 1:
            # 如果有多个地面图像，选择一个不同于原始地面图的图像
            other_indices = [idx for idx in ground_indices if idx != idx_ground]
            additional_idx_ground = random.choice(other_indices)
        else:
            # 如果只有一个地面图像，使用相同的图像（变换后会不同）
            additional_idx_ground = idx_ground
        additional_query_path = self.idx2ground_path[additional_idx_ground]
        additional_query_img = cv2.imread(additional_query_path)
        if additional_query_img is None:
            raise ValueError(f"Failed to load additional street image: {additional_query_path}")
        additional_query_img = cv2.cvtColor(additional_query_img, cv2.COLOR_BGR2RGB)
        additional_query_img = self.transforms_query2(image=additional_query_img)['image']


        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        query_img = self.transforms_query1(image=query_img)['image']
        # drone_img = self.transforms_drone(image=drone_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img,additional_query_img,reference_img1, reference_img2, label
    
    def __len__(self):
        return len(self.samples)



class U1652DatasetTrainGeomatchvigor_s(Dataset):
    def __init__(self, data_folder, use_vigor = True,transforms_query1=None,transforms_query2 = None, transforms_reference1=None, transforms_reference2=None, prob_flip=0.3, prob_rotate=0.3, shuffle_batch_size=128):


        if not isinstance(data_folder, str):
            raise ValueError(f"data_folder must be a string, got {type(data_folder)}")
        
        if use_vigor:
            street_files = sorted(glob.glob(os.path.join(data_folder, "train", "street", "*", "*.jpg")))

            vigor_g = sorted(glob.glob(os.path.join("/mnt/hdd/cky/hand_v/new_street/", "*", "*.png")))

            ground_files = street_files + vigor_g

            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        else:

            ground_files = sorted(glob.glob(os.path.join(data_folder, "train","street", "*", "*.jpg")))

            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
        
        if use_vigor:

            sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))

            vigor_s = sorted(glob.glob(os.path.join("/mnt/hdd/cky/hand_v/satellite/", "*", "*.png")))

            sat_files = sat_files+ vigor_s

        else:

            sat_files = sorted(glob.glob(os.path.join(data_folder, "train","satellite", "*", "*.jpg")))

        vigor_only = False
        if vigor_only == True:
            ground_files = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/street/", "*", "*_most_similar_block.jpg")))
            ground_ids = [os.path.basename(os.path.dirname(f)) for f in ground_files]
            sat_files = sorted(glob.glob(os.path.join("/mnt/hdd/cky/CKimage/satellite/", "*", "*.png")))

        

        if not ground_files:
            raise ValueError(f"No street images found in {os.path.join(data_folder, '/train/street')}")
        if not sat_files:
            raise ValueError(f"No satellite images found in {os.path.join(data_folder, '/train/satellite')}")

        df_sat = pd.DataFrame({
            "sat": [os.path.basename(os.path.dirname(f)) + '.jpg' for f in sat_files],
            "path": sat_files
        })

        df_ground = pd.DataFrame({
            "ground": [os.path.basename(f) for f in ground_files],
            "sat": [f"{id}.jpg" for id in ground_ids],
            "path_ground": ground_files,
            "location_id": ground_ids
        })


        sat2idx = {row["sat"]: i for i, row in df_sat.iterrows()}

        df_ground["sat"] = df_ground["sat"].map(sat2idx)

        

        unmapped_ground = df_ground["sat"].isna().sum()

        if unmapped_ground > 0:
            print(f"警告: {unmapped_ground} 张街景图像无法匹配卫星")
 
        self.df_ground = df_ground

        self.idx2ground_path = df_ground["path_ground"].to_dict()

        self.idx2sat_path = df_sat["path"].to_dict()
        self.idx2location = df_ground["location_id"].to_dict()
        self.idx2sat = df_sat["sat"].to_dict()

        self.location_to_ground_indices = df_ground.groupby("location_id").apply(lambda x: x.index.tolist()).to_dict()        

        self.pairs = [(idx, sat) for idx, sat in zip(self.df_ground.index, self.df_ground.sat.astype(int)) if not pd.isna(sat)]
        self.samples = []
        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2

        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.shuffle()

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        print("Initial pair_pool size:", len(pair_pool))
        if len(pair_pool) == 0:
            raise ValueError("No street-satellite pairs available. Check dataset loading.")
        random.shuffle(pair_pool)
        self.samples = pair_pool
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
    
    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        if query_img is None:
            raise ValueError(f"Failed to load street image: {self.idx2ground_path[idx_ground]}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
       

        location_id = self.idx2location[idx_ground]
        ground_indices = self.location_to_ground_indices[location_id]
        if len(ground_indices) > 1:

            other_indices = [idx for idx in ground_indices if idx != idx_ground]
            additional_idx_ground = random.choice(other_indices)
        else:

            additional_idx_ground = idx_ground
        additional_query_path = self.idx2ground_path[additional_idx_ground]
        additional_query_img = cv2.imread(additional_query_path)
        if additional_query_img is None:
            raise ValueError(f"Failed to load additional street image: {additional_query_path}")
        additional_query_img = cv2.cvtColor(additional_query_img, cv2.COLOR_BGR2RGB)
        additional_query_img = self.transforms_query2(image=additional_query_img)['image']

        
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        if reference_img is None:
            raise ValueError(f"Failed to load satellite image: {self.idx2sat_path[idx_sat]}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_img2 = reference_img.copy()
        
        query_img = self.transforms_query1(image=query_img)['image']

        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img2)['image']
        
        label = idx_sat
        return query_img,additional_query_img,reference_img1, reference_img2, label
    
    def __len__(self):
        return len(self.samples)


def select_google(data_folder=None,rand = 1.0):
    rand = 0.15
    test_ground_files = sorted(glob.glob(os.path.join(data_folder, "train", "google_ss", "*", "*.jpg")))
    if not test_ground_files:
        print(f"警告: 未在 {os.path.join(data_folder, 'train/google_ss')} 找到谷歌街景图像")

    ground_by_location = defaultdict(list)
    for f in test_ground_files:
        loc_id = os.path.basename(os.path.dirname(f))
        ground_by_location[loc_id].append(f)
    

    all_location_ids = list(ground_by_location.keys())
    num_select = max(1, int(len(all_location_ids) * rand))  
    selected_location_ids = random.sample(all_location_ids, num_select)
    print(f"谷歌数据集总 location_id: {len(all_location_ids)}, 选中的 location_id: {len(selected_location_ids)}")

    selected_ground_files = []

    for loc_id in selected_location_ids:
        selected_ground_files.extend(ground_by_location[loc_id])


    return selected_ground_files
