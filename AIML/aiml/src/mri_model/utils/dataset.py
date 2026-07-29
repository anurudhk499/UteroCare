from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


# ==========================================================
# CLASS MAPPING
# ==========================================================

CLASS_TO_IDX = {
    "adeno_mri": 0,
    "endometrial_cancer_mri": 1,
    "fibroid_mri": 2,
    "normal_uterus_MRI": 3,
}

IDX_TO_CLASS = {
    0: "Adenomyosis",
    1: "Endometrial_Cancer",
    2: "Fibroid",
    3: "Normal",
}


# ==========================================================
# DATASET
# ==========================================================

class UterusMRIDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        self._load_dataset()

    # ------------------------------------------------------

    def _load_dataset(self):

        for disease_folder in sorted(self.root_dir.iterdir()):

            if not disease_folder.is_dir():
                continue

            label = CLASS_TO_IDX[disease_folder.name]

            # --------------------------------------------------
            # Case 1 : Images directly inside folder (Normal)
            # --------------------------------------------------

            direct_images = list(disease_folder.glob("*.jpg"))
            direct_images += list(disease_folder.glob("*.jpeg"))
            direct_images += list(disease_folder.glob("*.png"))

            if len(direct_images) > 0:

                for img in direct_images:

                    self.samples.append((img, label))

                continue

            # --------------------------------------------------
            # Case 2 : Thickness folders
            # --------------------------------------------------

            for thickness_folder in disease_folder.iterdir():

                if not thickness_folder.is_dir():
                    continue

                for img in thickness_folder.iterdir():

                    if img.suffix.lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff",
                    ]:
                        continue

                    self.samples.append((img, label))

    # ------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # ------------------------------------------------------

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":


    from utils.transforms import train_transforms
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset = UterusMRIDataset(
        root_dir=BASE_DIR / "processed_dataset" / "train",
        transform=train_transforms,
    )

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print("Total Images :", len(dataset))

    image, label = dataset[0]

    print("Image Shape :", image.shape)
    print("Label :", label)
    print("Disease :", IDX_TO_CLASS[label])