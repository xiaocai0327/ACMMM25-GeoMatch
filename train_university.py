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
from geomatch.dataset.university import U1652DatasetEval, get_transforms_train_geomatch,U1652DatasetTrainGeomatch_5
from geomatch.utils import setup_system, Logger
from geomatch.trainer import train,train_contrast_geomatch,train_contrast_geomatch_5,train_contrast_geomatch_11
from geomatch.evaluate.university import evaluate
from geomatch.loss import InfoNCE
from geomatch.model import TimmModel
from geomatch.model import TimmModel_Geomatch,TimmModel_Geomatch_5

@dataclass
class Configuration:
    model: str = 'convnext_large.fb_in22k_ft_in1k_384'
    img_size: int = 512
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 2025
    epochs: int = 3
    batch_size: int = 12       # keep in mind real_batch_size = 2 * batch_size实际的批量大小（real batch size）是配置中定义的 batch_size 的两倍 
    verbose: bool = True
    gpu_ids: tuple = (0,1,2)           # GPU ids for training
    
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
    lr: float = 0.0005              # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
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

    checkpoint_start = None

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



if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join("/home/caikaiyan/geomatch-lx/", 'tensorboard'))
    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       time.strftime("%H%M%S"))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(__file__, os.path.join(model_path, "train.py"))
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))
    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))


    model = TimmModel_Geomatch_5(config.model,
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
    val_transforms,satellite_transforms, satellite_transforms_con, street_transforms, street_transforms_con = get_transforms_train_geomatch(
        img_size,
        img_size, 
        mean=mean, 
        std=std)

    train_dataset = U1652DatasetTrainGeomatch_5(
            data_folder=config.data_folder,  
            transforms_query = street_transforms,  
            transforms_drone = street_transforms_con,  
            transforms_reference1=satellite_transforms, 
            transforms_reference2=satellite_transforms_con,  
            prob_flip=config.prob_flip,  
            prob_rotate=config.prob_rotate,  
            shuffle_batch_size=config.batch_size,  
            )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=config.custom_sampling,
                                  pin_memory=True)
    
    # Query Ground Images TestReference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Reference Satellite Images
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
        

        train_loss = train_contrast_geomatch_11(config,
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