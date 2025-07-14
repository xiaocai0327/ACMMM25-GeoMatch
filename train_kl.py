import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from geomatch.dataset.university import get_transforms_train_geomatch
from geomatch.dataset.university import U1652DatasetEval
class ConvNextProbabilisticEmbedding(nn.Module):
    def __init__(self, model_name="convnext_large", embed_dim=512, weight_path=None):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        if weight_path:
            state_dict = torch.load(weight_path)
            self.base_model.load_state_dict(state_dict, strict=False)
        feature_dim = self.base_model(torch.randn(1, 3, 224, 224)).shape[1]
        print(f"feature_dim: {feature_dim}")
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.mu_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        for layer in self.logvar_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        features = self.base_model(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features) + 1e-6
        
        return mu, logvar
class DistributionKLDivergence(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu1, logvar1, mu2, logvar2):
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        term1 = logvar2 - logvar1
        term2 = var1 / var2
        term3 = (mu1 - mu2).pow(2) / var2
        term4 = -1
        kl = 0.5 * (term1 + term2 + term3 + term4)
        return kl.sum(dim=1).mean()
class UniversityDistributionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div = DistributionKLDivergence()
    
    def forward(self, mu, logvar):
        batch_size = mu.size(0)
        total_loss = 0.0
        count = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    total_loss += self.kl_div(
                        mu[i].unsqueeze(0), 
                        logvar[i].unsqueeze(0),
                        mu[j].unsqueeze(0),
                        logvar[j].unsqueeze(0)
                    )
                    count += 1

        return total_loss / batch_size
def train_university_distribution(data_dir, save_dir, config):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "university_ground_embedder.pth")
    
    img_size = (config['image_size'], config['image_size'])
    val_transforms,_, _, _, _ = get_transforms_train_geomatch(
        img_size,
        img_size, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])

    dataset = U1652DatasetEval(data_folder=data_dir,
                                               mode="query",
                                               transforms=val_transforms,
                                               )

    data_loader = DataLoader(dataset,
                                       batch_size=8,
                                       num_workers=8,
                                       shuffle=True,
                                       pin_memory=True)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNextProbabilisticEmbedding(
        model_name=config['model_name'],
        embed_dim=config['embed_dim'],
        weight_path=config['weight_path']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    criterion = UniversityDistributionLoss().to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )

    print(f"Train start, with{config['epochs']} epoch.")
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(device)
            mu, logvar = model(images)
            loss = criterion(mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % config['log_interval'] == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Batch {batch_idx}/{len(data_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        scheduler.step(avg_epoch_loss)
        
        print(f"Epoch {epoch+1} finish |avgloss: {avg_epoch_loss:.4f}")
        save_path = os.path.join(save_dir, 'weights_e{}_{:.4f}.pth'.format(epoch, avg_epoch_loss ))
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"save model to {save_path}")
    
    print("Train over!")
    return model, best_loss


if __name__ == "__main__":

    config = {

        'model_name': "convnext_large",
        'weight_path': "/mnt/hdd/cky/convnext_large_weights/pytorch_model.bin",
        'embed_dim': 512,
        'epochs': 2,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'log_interval': 50,
        'num_workers': 8,
        'image_size': 512
    }
    university_dir = '/mnt/hdd/cky/dataset/University-Release/train/street/'
    model_save_path = "/mnt/hdd/cky/geolocalization_models"
    model, best_loss = train_university_distribution(
        data_dir=university_dir,
        save_dir=model_save_path,
        config=config
    )
    
    print(f"train over,best_loss: {best_loss:.4f}")