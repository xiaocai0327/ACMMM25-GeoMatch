# GeoMatch: Multi-View Contrastive Learning for Limited CVGL with Semantic Uncertainty

Solution for ACMMM25 Multimedia Street Satellite Matching Challenge In Multiple-environment

Cross-view geolocalization (CVGL) remains a challenging task due to the drastic viewpoint variations and occlusions between ground and aerial image pairs, especially under limited field-of-view (FoV) conditions in real-world ground images. In this work, we propose a novel multi-view contrastive learning framework that significantly enhances ground-to-aerial image matching with limited FoV by integrating cross-dataset expansion and multi-view contrastive learning.
To obtain more training samples, we use a probabilistic embedding mechanism based on a Gaussian distribution to model image semantic uncertainty and select image pairs from VIGOR semantically aligned with University-1652, constructing a high-quality auxiliary dataset, VIGOR<sub>aux</sub>. Furthermore, we design a unified contrastive learning objective that combines single-view and cross-view losses, incorporating drone-view images as intermediaries to bridge the semantic gap between ground and aerial views. Experiments on the University-1652 benchmark demonstrate superior performance. 

## Usage

### 1. Auxiliary Sample Collection

#### Cosine Similarity-based Data Select
```bash
python get_topk_cos.py
```
#### KL Divergence Training
```bash
python train_kl.py
```

#### KL Divergence-based Data Select
```bash
python get_topk_kl.py
```

### 2. Model Training

#### Training on University-1652 or Pre-training VIGOR<sub>aux</sub>
```bash
python train_university.py
```

#### Pre-training on VIGOR (Ablation Study)
```bash
python train_geomatch_vigor.py
```
