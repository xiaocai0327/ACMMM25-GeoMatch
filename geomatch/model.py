import torch
import timm
import numpy as np
import torch.nn as nn
import random
from torchvision.transforms import Resize

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:

            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
             
            return image_features

class TimmModel_aug(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel_aug, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
            imgq1 = img1
            start = int(imgq1.size(-1)*70/360)
            stop = imgq1.size(-1)
            fov = random.randint(start, stop)
            imgq2 = imgq1[...,:fov]
            image_features1 = self.model(imgq2)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2              
        else:
            image_features = self.model(img1)
             
            return image_features


import os

class TimmModel_Geomatch(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 random_fov=False):

        super(TimmModel_Geomatch, self).__init__()
        
        self.img_size = img_size
        self.random_fov = random_fov
        

        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)

        weight_path = "/mnt/hdd/cky/convnext_large_weights/pytorch_model.bin"
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict, strict=False)

        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.reducer = nn.Linear(4096, 1024)
        


    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, imgq1, imgq2=None, imgr1=None, imgr2=None):
        return_feature_map=False
        if imgq2 is not None:
            if self.random_fov == False:
                if return_feature_map:
                    image_featuresq1 = self.model.forward_features(imgq1)
                    image_featuresq1 = get_lpn_features(image_featuresq1, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresq1 = self.reducer(image_featuresq1 )
                    image_featuresq2 = self.model.forward_features(imgq2)
                    image_featuresq2 = get_lpn_features(image_featuresq2, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresq2 = self.reducer(image_featuresq2 )
                    image_featuresr1 = self.model.forward_features(imgr1)
                    image_featuresr1 = get_lpn_features(image_featuresr1, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresr1 = self.reducer( image_featuresr1)
                    image_featuresr2 = self.model.forward_features(imgr2)
                    image_featuresr2 = get_lpn_features(image_featuresr2, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresr2 = self.reducer(image_featuresr2 )
                else:
                    image_featuresq1 = self.model(imgq1)     
                    image_featuresq2 = self.model(imgq2)
                    image_featuresr1 = self.model(imgr1)
                    image_featuresr2 = self.model(imgr2)
                return image_featuresq1, image_featuresq2, image_featuresr1, image_featuresr2         
            
            else:
                # random fov between 70-360
                random_fov = random.randint(int(imgq2.size(-1)*7/36), imgq2.size(-1))
                imgq2 = imgq2[...,:random_fov]
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2)
                
                return image_featuresq1, image_featuresq2, image_featuresr1,  image_featuresr2          
              
        else:
            
            image_features = self.model(imgq1)
             
            return image_features





class TimmModel_Geomatch_5(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 random_fov=False):

        super(TimmModel_Geomatch_5, self).__init__()
        
        self.img_size = img_size
        self.random_fov = random_fov
        

        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        use_base = False
        if use_base == True:
            weight_path = "/mnt/hdd/cky/convnext_base_weights/pytorch_model.bin"
        else:
            weight_path = "/mnt/hdd/cky/convnext_large_weights/pytorch_model.bin"
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict, strict=False)

        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.reducer = nn.Linear(4096, 1024)
        

    
    def forward_query(self, query_img):

        return self.model(query_img)

    def forward_reference(self, reference_img):

        return self.model(reference_img)


    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, imgq1, imgq2=None, imgr1=None, imgr2=None,imgq3=None):
        return_feature_map=False
        if imgq2 is not None:
            if self.random_fov == False:
                if return_feature_map:
                    image_featuresq1 = self.model.forward_features(imgq1)
                    image_featuresq1 = get_lpn_features(image_featuresq1, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresq1 = self.reducer(image_featuresq1 )
                    image_featuresq2 = self.model.forward_features(imgq2)
                    image_featuresq2 = get_lpn_features(image_featuresq2, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresq2 = self.reducer(image_featuresq2 )
                    image_featuresr1 = self.model.forward_features(imgr1)
                    image_featuresr1 = get_lpn_features(image_featuresr1, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresr1 = self.reducer( image_featuresr1)
                    image_featuresr2 = self.model.forward_features(imgr2)
                    image_featuresr2 = get_lpn_features(image_featuresr2, num_parts=4, pool='avg', no_overlap=True)
                    image_featuresr2 = self.reducer(image_featuresr2 )
                else:
                    image_featuresq1 = self.model(imgq1)     
                    image_featuresq2 = self.model(imgq2)
                    image_featuresq3 = self.model(imgq3)
                    image_featuresr1 = self.model(imgr1)
                    image_featuresr2 = self.model(imgr2)
                return image_featuresq1, image_featuresq2, image_featuresr1, image_featuresr2  ,image_featuresq3           
            
            else:
                # random fov between 70-360
                random_fov = random.randint(int(imgq2.size(-1)*7/36), imgq2.size(-1))
                imgq2 = imgq2[...,:random_fov]
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2)
                
                return image_featuresq1, image_featuresq2, image_featuresr1,  image_featuresr2      
              
        else:
            
            image_features = self.model(imgq1)
             
            return image_features

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA
def get_lpn_features(x, num_parts=4, pool='avg', no_overlap=True):
    """
    Process a ConvNeXt feature map into 4 parts using LPN method and concatenate them.
    
    Args:
        x (torch.Tensor): Input feature map of shape [B, C, H, W]
        num_parts (int): Number of parts to divide into, default is 4
        pool (str): Pooling type, 'avg' or 'max', default is 'avg'
        no_overlap (bool): Whether to remove overlap between parts, default is True
    
    Returns:
        torch.Tensor: Concatenated feature vector of shape [B, 4*C]
    """
    result = []
    if pool == 'avg':
        pooling = nn.AdaptiveAvgPool2d((1, 1))
    elif pool == 'max':
        pooling = nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise ValueError("Pool must be 'avg' or 'max'")

    B, C, H, W = x.size()
    c_h, c_w = int(H / 2), int(W / 2)
    per_h, per_w = H / (2 * num_parts), W / (2 * num_parts)

    # Resize feature map if too small
    if per_h < 1 and per_w < 1:
        new_H = H + (num_parts - c_h) * 2
        new_W = W + (num_parts - c_w) * 2
        x = F.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
        H, W = new_H, new_W
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * num_parts), W / (2 * num_parts)

    per_h, per_w = math.floor(per_h), math.floor(per_w)

    for i in range(num_parts):
        i = i + 1
        if i < num_parts:
            # Extract current part
            x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
            if no_overlap and i > 1:
                # Subtract previous part to avoid overlap
                x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h), (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                x_curr = x_curr - x_pad
            pooled = pooling(x_curr)
            result.append(pooled)
        else:
            # Last part: entire feature map
            if no_overlap and i > 1:
                x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h), (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                pad_h = c_h - (i - 1) * per_h
                pad_w = c_w - (i - 1) * per_w
                if x_pre.size(2) + 2 * pad_h == H:
                    x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                else:
                    ep = H - (x_pre.size(2) + 2 * pad_h)
                    x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                x = x - x_pad
            pooled = pooling(x)
            result.append(pooled)

    # Concatenate and reshape to [B, 4*C]
    features = torch.cat(result, dim=1)  # [B, 4*C, 1, 1]
    features = features.view(B, -1)  # [B, 4*C]
    return features