import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from data import get_dataloaders
from model.R50_ViT import R50ViT
from model.ResNet50 import ResNet50
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

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        _, preds = torch.max(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_acc, epoch_macro_f1


def evaluate(
    model,
    data_loader,
    criterion,
    device,
    class_names,
    mode="Validation",
    print_report=False
):
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

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    if print_report and mode=="Testing":
        print(f"\n{mode} classification report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=class_names,
                digits=4,
                zero_division=0
            )
        )

    return epoch_loss, epoch_acc, epoch_macro_f1


def save_training_log(logs, save_path):
    df = pd.DataFrame(logs)
    df.to_csv(save_path, index=False)


def plot_metrics(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "loss_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "accuracy_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_macro_f1"], label="Train Macro-F1")
    plt.plot(df["epoch"], df["val_macro_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and Validation Macro-F1")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "macro_f1_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def build_model(args, num_classes):
    model_name = args.model.lower()

    if model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)

        model_config = {
            "model_name": "ResNet50"
        }

    elif model_name == "vit":
        model = VisionTransformer(
            num_channels=args.num_channels,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            img_size=args.img_size,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            transformer_units=args.transformer_units,
            num_classes=num_classes,
            dropout=args.dropout
        )

        model_config = {
            "model_name": "VisionTransformer",
            "num_channels": args.num_channels,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "mlp_dim": args.mlp_dim,
            "transformer_units": args.transformer_units,
            "dropout": args.dropout
        }

    elif model_name == "r50vit":
        model = R50ViT(
            num_classes=num_classes,
            img_size=args.img_size,
            downsample_ratio=args.downsample_ratio,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            transformer_units=args.transformer_units,
            dropout=args.dropout
        )

        model_config = {
            "model_name": "R50ViT",
            "downsample_ratio": args.downsample_ratio,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "mlp_dim": args.mlp_dim,
            "transformer_units": args.transformer_units,
            "dropout": args.dropout
        }

    else:
        raise ValueError(
            "Unsupported model. Please choose one of: resnet50, vit, r50vit"
        )

    return model, model_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50, Vision Transformer, or R50+ViT for brain tumor classification."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["resnet50", "vit", "r50vit"],
        help="Model to train."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Epic_and_CSCR_hospital_Dataset",
        help="Path to dataset folder."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder. If not provided, it will be created based on model name."
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--print_val_report",
        action="store_true",
        help="Print classification report for validation after every epoch."
    )

    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=192,
        help="ViT-Tiny: 192, ViT-Small: 384, ViT-Base: 768."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=3,
        help="ViT-Tiny: 3, ViT-Small: 6, ViT-Base: 12."
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=768,
        help="ViT-Tiny: 768, ViT-Small: 1536, ViT-Base: 3072."
    )
    parser.add_argument(
        "--transformer_units",
        type=int,
        default=12,
        help="Number of Transformer encoder blocks."
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=16,
        choices=[16, 32],
        help="R50+ViT-B/16 uses 16; R50+ViT-B/32 uses 32."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs_{args.model}"

    os.makedirs(args.output_dir, exist_ok=True)

    best_model_path = os.path.join(
        args.output_dir,
        f"best_{args.model}_model.pth"
    )
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers
    )

    print("Classes:", class_names)
    print("Number of classes:", num_classes)

    model, model_config = build_model(args, num_classes)
    model = model.to(device)

    print("Model:", model_config["model_name"])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    best_val_macro_f1 = -1.0
    patience_counter = 0
    logs = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")

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
            mode="Validation",
            print_report=args.print_val_report
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
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "class_names": class_names,
                "num_classes": num_classes,
                "img_size": args.img_size,
                **model_config
            }

            torch.save(checkpoint, best_model_path)

            print(f"Saved best model with Val Macro-F1: {best_val_macro_f1:.4f}")

        else:
            patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    plot_metrics(
        csv_path=log_csv_path,
        output_dir=args.output_dir
    )

    print(f"\nSaved training log and curves to: {args.output_dir}")
    print("\nLoading best model for testing...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        mode="Testing",
        print_report=True
    )

    print("\nFinal Test Result")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
