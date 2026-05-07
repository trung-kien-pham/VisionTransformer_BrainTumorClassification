# Vision Transformer for Brain Tumor Classification

A PyTorch implementation of a custom **Vision Transformer (ViT)** model for brain tumor MRI image classification on the **[Brain Tumor MRI Dataset](https://data.mendeley.com/datasets/zwr4ntf94j/6)**.

This project trains a Vision Transformer from scratch to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Structure

```text
Vision-Transformer-Brain-Tumor-Classification/
├── model/
│   └── ViT.py
├── data.py
├── train.py
├── requirements.txt
└── README.md
```

## Dataset Structure

The dataset should be organized as follows:

```text
Epic and CSCR hospital Dataset/
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

Each subfolder contains MRI images belonging to one class.

## Model

The Vision Transformer model is implemented in:

```text
model/ViT.py
```

The custom ViT architecture includes:

- Patch embedding using `Conv2d`
- Learnable class token
- Learnable positional embedding
- Transformer encoder blocks
- Multi-head self-attention
- MLP block
- Classification head


## Training Pipeline

The training script is implemented in:

```text
train.py
```

The training pipeline includes:

- Training loop
- Validation loop
- Cross-entropy loss
- AdamW optimizer
- Cosine annealing learning rate scheduler
- Accuracy calculation
- Macro-F1 score calculation
- Per-class F1 score during validation
- CSV logging
- Training curve visualization

## Metrics

The following metrics are used to evaluate the model:

- Loss
- Accuracy
- Macro-F1 score

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

## How to Train

Run the training script:

```bash
python train.py
```

Before training, make sure the dataset path in `train.py` is correct:

```python
data_dir = "Path/to/dataset"
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

The file `training_log.csv` stores the training and validation metrics for each epoch.

The figures show:

- Training and validation loss
- Training and validation accuracy
- Training and validation macro-F1 score

## Notes

This project uses a custom Vision Transformer implementation for learning and experimental purposes.

For small or medium-sized datasets, training ViT from scratch can be challenging. Using a smaller ViT configuration such as ViT-T/16 or ViT-S/16 is recommended before trying larger models such as ViT-B/16.

## License

This project is released under the MIT License.
