import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from congeo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms_train_congeo,U1652DatasetTrainConGeo,U1652DatasetTrainConGeo_5,U1652DatasetTrainConGeo_5_google
from congeo.utils import setup_system, Logger
from congeo.trainer import train,train_contrast_congeo,train_contrast_congeo_5
from congeo.evaluate.university import evaluate
from congeo.loss import InfoNCE
from congeo.model import TimmModel
from congeo.model import TimmModel_ConGeo,TimmModel_ConGeo_5

@dataclass
class Configuration:
    
    # Model
    # model: str = 'convnext_large.fb_in22k_ft_in1k_384'
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 2025
    epochs: int = 3
    batch_size: int = 60# keep in mind real_batch_size = 2 * batch_size实际的批量大小（real batch size）是配置中定义的 batch_size 的两倍
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3,4)           # GPU ids for training
    
    # Eval
    batch_size_eval: int = 100
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0001                # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"           # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/mnt/hdd/cky/dataset/University-Release"
    
    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously
    prob_rotate:float=0.0
    # Savepath for model checkpoints
    model_path: str = "/mnt/hdd/cky/checkpoint_univisity/"
    
    # Eval before training
    zero_shot: bool = False 
    # zero_shot: bool = True

    # Checkpoint to start from

    # checkpoint_start = None
    checkpoint_start = "/mnt/hdd/cky/checkpoints_90/weights_else_ceshide32_63.9654.pth"

#   /mnt/hdd/lx/checkpoints/convnext_base_22k_1k_384.pt
    # set num_workers to 0 if on Windows
    num_workers: int = 32
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 





if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/mnt/hdd/cky/dataset/University-Release/train/satellite/'
    config.gallery_folder_train = '/mnt/hdd/cky/dataset/University-Release/train/street/'   
    config.query_folder_test = '/mnt/hdd/cky/dataset/University-Release/test/query_street/' 
    config.gallery_folder_test = '/mnt/hdd/cky/dataset/University-Release/test/gallery_satellite/'
# elif config.dataset == 'U1652-S2D':
#     config.query_folder_train = './data/U1652/train/satellite'
#     config.gallery_folder_train = './data/U1652/train/drone'    
#     config.query_folder_test = './data/U1652/test/query_satellite'
#     config.gallery_folder_test = './data/U1652/test/gallery_drone'


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,4,5,6"
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join("/home/caikaiyan/ConGeo-lx/", 'tensorboard'))
    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(__file__, os.path.join(model_path, "train.py"))


    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#




    print("\nModel: {}".format(config.model))


    model = TimmModel_ConGeo_5(config.model,
                          pretrained=False,
                          img_size=config.img_size)
    
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # change_ft
    print("冻结 Stage 1、Stage 2 的参数...")
    for name, param in model.named_parameters():
        if 'stages.0' in name or 'stages.1' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # print("参数冻结状态：")
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")


    # 可选：统计可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数数量: {trainable_params}")
    print(f"总参数数量: {total_params}")
    print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")


    # Model to device   
    model = model.to(config.device)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            


    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms satellite_transforms, satellite_transforms_con, street_transforms, street_transforms_con
    val_transforms,satellite_transforms, satellite_transforms_con, street_transforms, street_transforms_con = get_transforms_train_congeo(img_size,img_size, mean=mean, std=std)
                    #   transforms_query=None, transforms_drone=None, transforms_reference1=None, transforms_reference2=None,                                                                                                           
    # Train
    # train_dataset = U1652DatasetTrainConGeo(
    #         data_folder=config.data_folder,  # University-1652 根目录，如 "/mnt/hdd/cky/dataset/University-Release/"
    #         transforms_query1=street_transforms,  # 街景变换 (query_img1)
    #         transforms_query2=street_transforms_con,  # 增强街景变换 (query_img2)
    #         transforms_reference1=satellite_transforms,  # 卫星变换 (reference_img1)
    #         transforms_reference2=satellite_transforms_con,  # 增强卫星变换 (reference_img2)
    #         prob_flip=config.prob_flip,  # 翻转概率，如 0.5
    #         prob_rotate=config.prob_rotate,  # 旋转概率，如 0.3
    #         shuffle_batch_size=config.batch_size  # 批次大小，如 64
    #         )
    
    # train_dataset = U1652DatasetTrainConGeo(
    #         data_folder=config.data_folder,  # University-1652 根目录，如 "/mnt/hdd/cky/dataset/University-Release/"
    #         transforms_query=street_transforms,  # 街景变换 (query_img1)
    #         transforms_drone=street_transforms_con,  # 增强街景变换 (query_img2)
    #         transforms_reference1=satellite_transforms,  # 卫星变换 (reference_img1)
    #         transforms_reference2=satellite_transforms_con,  # 增强卫星变换 (reference_img2)
    #         prob_flip=config.prob_flip,  # 翻转概率，如 0.5
    #         prob_rotate=config.prob_rotate,  # 旋转概率，如 0.3
    #         shuffle_batch_size=config.batch_size  # 批次大小，如 64
    #         )
    train_dataset = U1652DatasetTrainConGeo_5_google(
            data_folder=config.data_folder,  # University-1652 根目录，如 "/mnt/hdd/cky/dataset/University-Release/"
            transforms_query=street_transforms,  # 街景变换 (query_img1)
            transforms_drone=street_transforms_con,  # 增强街景变换 (query_img2)
            transforms_reference1=satellite_transforms,  # 卫星变换 (reference_img1)
            transforms_reference2=satellite_transforms_con,  # 增强卫星变换 (reference_img2)
            prob_flip=config.prob_flip,  # 翻转概率，如 0.5
            prob_rotate=config.prob_rotate,  # 旋转概率，如 0.3
            shuffle_batch_size=config.batch_size  # 批次大小，如 64
            )
    

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                # shuffle = True,
                                  pin_memory=True)
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = torch.amp.GradScaler('cuda', init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#
    
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        

        train_loss = train_contrast_congeo_5(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            
            writer.add_scalar('Recall/R@1', r1_test, epoch)
    
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                    print("save at",'{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test*10))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                    print("save at",'{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test*10))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
    writer.close()