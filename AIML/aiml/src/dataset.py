import os
import random

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from config import *

# ==========================================
# DATASET CLASS
# ==========================================

class UterusDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform

        self.images = []
        self.labels = []

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(CLASS_NAMES)
        }

        for cls in CLASS_NAMES:

            class_path = os.path.join(root_dir, cls)

            if not os.path.exists(class_path):
                continue

            for img in os.listdir(class_path):

                self.images.append(
                    os.path.join(class_path, img)
                )

                self.labels.append(
                    self.class_to_idx[cls]
                )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.transform:

            image = self.transform(image)

        return image, label


# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(5),

    transforms.RandomAffine(

        degrees=0,

        translate=(0.03,0.03),

        scale=(0.95,1.05)

    ),

    transforms.ColorJitter(

        brightness=0.08,

        contrast=0.08

    ),

    transforms.Resize(

        (IMAGE_SIZE, IMAGE_SIZE)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    ),

    transforms.RandomErasing(

        p=0.25,

        scale=(0.02,0.08)

    )

])



val_transform = transforms.Compose([

    transforms.Resize(

        (IMAGE_SIZE, IMAGE_SIZE)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


# ==========================================
# DATALOADERS
# ==========================================

def get_dataloaders():

    train_dataset = UterusDataset(

        "../preprocessed_dataset_v2/train",

        transform=train_transform

    )

    val_dataset = UterusDataset(

        "../preprocessed_dataset_v2/val",

        transform=val_transform

    )

    test_dataset = UterusDataset(

        "../preprocessed_dataset_v2/test",

        transform=val_transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )

    print(f"\nTraining Images : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images : {len(test_dataset)}")

    return train_loader, val_loader, test_loader