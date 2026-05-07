import os
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from data import get_dataloaders
from model.ViT import VisionTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []

    loop = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_macro_f1

def evaluate(model, data_loader, criterion, device, class_names, mode="Validation"):
    model.eval()

    running_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []

    loop = tqdm(data_loader, desc=mode, leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)

            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro")

    if mode == "Testing":
        print(f"\n{mode} F1-score per class:")
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print(report)

    return epoch_loss, epoch_acc, epoch_macro_f1

def save_training_log(logs, save_path):
    df = pd.DataFrame(logs)
    df.to_csv(save_path, index=False)
    print(f"Saved training log to: {save_path}")

def plot_metrics(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_macro_f1"], label="Train Macro-F1")
    plt.plot(df["epoch"], df["val_macro_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and Validation Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "macro_f1_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved metric plots to: {output_dir}")

def main():
    data_dir = "Epic and CSCR hospital Dataset"
    output_dir = "outputs_vit"

    os.makedirs(output_dir, exist_ok=True)

    img_size = 224
    patch_size = 16
    num_channels = 3

    batch_size = 1
    num_epochs = 1
    learning_rate = 1e-4
    val_ratio = 0.2

    embed_dim = 192         # ViT-Tiny: 192,  ViT-Small: 384,  ViT-Base: 768
    num_heads = 3           # ViT-Tiny: 3,    ViT-Small: 6,    ViT-Base: 12
    mlp_dim = 786           # ViT-Tiny: 768,  ViT-Small: 1536, ViT-Base: 3072
    transformer_units = 12  # ViT-Tiny: 12,   ViT-Small: 12,   ViT-Base: 12
    dropout = 0.1

    best_model_path = os.path.join(output_dir, "best_vit_model.pth")
    log_csv_path = os.path.join(output_dir, "train.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=0
    )

    print("Classes:", class_names)
    print("Number of classes:", num_classes)

    model = VisionTransformer(
        num_channels=num_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        img_size=img_size,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        transformer_units=transformer_units,
        num_classes=num_classes,
        dropout=dropout
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    best_val_macro_f1 = 0.0
    logs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc, val_macro_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            mode="Validation"
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro-F1: {train_macro_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "learning_rate": current_lr
        })

        save_training_log(logs, log_csv_path)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_names": class_names,
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "mlp_dim": mlp_dim,
                    "transformer_units": transformer_units,
                    "num_classes": num_classes
                },
                best_model_path
            )

            print(f"Saved best model with Val Macro-F1: {best_val_macro_f1:.4f}")

    plot_metrics(
        csv_path=log_csv_path,
        output_dir=output_dir
    )

    print("\nLoading best model for testing...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        mode="Testing"
    )

    print("\nFinal Test Result")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")

if __name__ == "__main__":
    main()
    