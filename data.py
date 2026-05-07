import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform

def get_dataloaders(
    data_dir,
    img_size=224,
    batch_size=16,
    val_ratio=0.2,
    num_workers=0
):
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    train_transform, test_transform = get_transforms(img_size)

    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    class_names = full_train_dataset.classes
    num_classes = len(class_names)

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_names, num_classes

if __name__ == "__main__":
    data_dir = "Epic and CSCR hospital Dataset"
    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(data_dir)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    