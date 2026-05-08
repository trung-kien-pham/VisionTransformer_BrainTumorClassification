# Vision Transformer for Brain Tumor Classification

A PyTorch implementation of deep learning models for brain tumor MRI image classification on the [Brain Tumor MRI Dataset](https://data.mendeley.com/datasets/zwr4ntf94j/6).

This project supports three model choices:

- **ResNet50**
- **Vision Transformer (ViT)**
- **Hybrid R50+ViT**

The models are trained to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Structure

```text
Vision-Transformer-Brain-Tumor-Classification/
├── Epic_and_CSCR_hospital_Dataset/
├── model/
│   ├── R50_ViT.py
│   ├── ResNet50.py
│   └── ViT.py
├── data.py
├── train.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset Structure

The dataset should be organized as follows:

```text
Epic_and_CSCR_hospital_Dataset/
├── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each class folder contains MRI images belonging to that class.

## Models

### 1. ResNet50

The ResNet50 model is implemented in:

```text
model/ResNet50.py
```

This model uses bottleneck residual blocks and is used as a CNN baseline for brain tumor classification.

### 2. Vision Transformer

The custom Vision Transformer is implemented in:

```text
model/ViT.py
```

The ViT architecture includes:

- Patch embedding using `Conv2d`
- Learnable class token
- Learnable positional embedding
- Transformer encoder blocks
- Multi-head self-attention
- MLP block
- Classification head

Example ViT-Tiny style configuration:

```python
embed_dim = 192
num_heads = 3
mlp_dim = 768
transformer_units = 12
patch_size = 16
```

### 3. R50+ViT

The hybrid R50+ViT model is implemented in:

```text
model/R50_ViT.py
```

This model uses a ResNet50V2-style backbone to extract CNN feature maps before sending them to a Transformer encoder.

The general pipeline is:

```text
Input MRI image
→ ResNet50V2 backbone
→ CNN feature map
→ Flatten spatial feature map into tokens
→ Linear projection
→ Transformer encoder
→ Classification head
```

The hybrid model supports two downsampling settings:

```text
R50+ViT-B/16 style: downsample_ratio = 16
R50+ViT-B/32 style: downsample_ratio = 32
```

## Training Pipeline

The unified training script is:

```text
train.py
```

The training pipeline includes:

- Training loop
- Validation loop
- Testing loop
- Cross-entropy loss
- AdamW optimizer
- Cosine annealing learning rate scheduler
- Accuracy calculation
- Macro-F1 score calculation
- Classification report
- CSV metric logging
- Training curve visualization
- Best model checkpoint saving based on validation macro-F1
- Optional early stopping with patience

## Metrics

The following metrics are used:

- Loss
- Accuracy
- Macro-F1 score
- Per-class precision, recall, and F1-score in the classification report

Macro-F1 is useful when the number of samples between classes is imbalanced, because it gives equal importance to each class.

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/Vision-Transformer-Brain-Tumor-Classification.git
cd Vision-Transformer-Brain-Tumor-Classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The main libraries used in this project are:

```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
tqdm
Pillow
```

## How to Train

The file `train.py` allows training different models using the `--model` argument.

### Train ResNet50

```bash
python train.py --model resnet50
```

### Train Vision Transformer

```bash
python train.py --model vit
```

Example with ViT-Tiny style settings:

```bash
python train.py --model vit --embed_dim 192 --num_heads 3 --mlp_dim 768 --transformer_units 12
```

Example with ViT-Small style settings:

```bash
python train.py --model vit --embed_dim 384 --num_heads 6 --mlp_dim 1536 --transformer_units 12
```

Example with ViT-Base style settings:

```bash
python train.py --model vit --embed_dim 768 --num_heads 12 --mlp_dim 3072 --transformer_units 12
```

### Train R50+ViT

For R50+ViT-B/16 style:

```bash
python train.py --model r50vit --downsample_ratio 16
```

For R50+ViT-B/32 style:

```bash
python train.py --model r50vit --downsample_ratio 32
```

## Common Training Arguments

You can modify common training settings from the command line:

```bash
python train.py --model vit --batch_size 8 --num_epochs 50 --learning_rate 1e-4
```

Useful arguments include:

```text
--model              Choose model: resnet50, vit, or r50vit
--data_dir           Path to dataset folder
--output_dir         Folder to save results
--img_size           Input image size
--batch_size         Batch size
--num_epochs         Number of training epochs
--learning_rate      Learning rate
--weight_decay       Weight decay
--val_ratio          Validation split ratio
--num_workers        Number of dataloader workers
--patience           Early stopping patience
```

Before training, make sure the dataset path is correct. For example:

```bash
python train.py --model vit --data_dir Epic_and_CSCR_hospital_Dataset
```

## Output Files

After training, the output folder may contain:

```text
outputs_vit/
├── best_vit_model.pth
├── training_log.csv
├── loss_curve.png
├── accuracy_curve.png
└── macro_f1_curve.png
```

For other models, the output folder may be:

```text
outputs_resnet50/
outputs_r50vit/
```

The file `training_log.csv` stores training and validation metrics for each epoch.

The generated figures show:

- Training and validation loss
- Training and validation accuracy
- Training and validation macro-F1 score

## Notes

This project is designed for learning and experimentation with CNNs, Vision Transformers, and hybrid CNN-Transformer models for medical image classification.

Training ViT from scratch can be challenging on small or medium-sized datasets. Therefore, smaller configurations such as ViT-T/16 or ViT-S/16 are recommended before trying larger models such as ViT-B/16.

For the hybrid R50+ViT model, `downsample_ratio=16` keeps a larger token grid than `downsample_ratio=32`, which may preserve more spatial information for MRI image classification.