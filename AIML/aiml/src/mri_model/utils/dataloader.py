from torch.utils.data import DataLoader

from configs.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from utils.dataset import UterusMRIDataset
from utils.transforms import (
    train_transforms,
    val_transforms,
    test_transforms,
)


def get_dataloaders():

    train_dataset = UterusMRIDataset(
        root_dir=TRAIN_DIR,
        transform=train_transforms,
    )

    val_dataset = UterusMRIDataset(
        root_dir=VAL_DIR,
        transform=val_transforms,
    )

    test_dataset = UterusMRIDataset(
        root_dir=TEST_DIR,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )