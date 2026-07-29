import random
import shutil
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

# ==========================================================
# SETTINGS
# ==========================================================

random.seed(42)

SOURCE = Path("../uterus_dataset")
DESTINATION = Path("../processed_dataset")

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================================
# CLASS FOLDERS
# ==========================================================

CLASS_MAPPING = {
    "adeno_mri": "Adenomyosis",
    "fibroid_mri": "Fibroid",
    "endometrial_cancer_mri": "Endometrial_Cancer",
    "normal_uterus_MRI": "Normal"
}

# ==========================================================

if DESTINATION.exists():
    shutil.rmtree(DESTINATION)

print("=" * 70)
print("PREPARING DATASET")
print("=" * 70)

# ==========================================================
# COLLECT ALL IMAGES
# ==========================================================

images = []
labels = []
relative_paths = []

for folder in CLASS_MAPPING.keys():

    disease_folder = SOURCE / folder

    # -----------------------------------------
    # Normal folder
    # -----------------------------------------

    direct_images = [
        f for f in disease_folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if len(direct_images) > 0:

        for img in direct_images:

            images.append(img)
            labels.append(folder)
            relative_paths.append(img.name)

        continue

    # -----------------------------------------
    # Thickness folders
    # -----------------------------------------

    for thickness in disease_folder.iterdir():

        if not thickness.is_dir():
            continue

        for img in thickness.iterdir():

            if img.suffix.lower() not in VALID_EXTENSIONS:
                continue

            images.append(img)
            labels.append(folder)

            relative_paths.append(
                str(Path(thickness.name) / img.name)
            )

# ==========================================================
# STRATIFIED SPLIT
# ==========================================================

train_imgs, temp_imgs, train_labels, temp_labels, train_paths, temp_paths = train_test_split(
    images,
    labels,
    relative_paths,
    train_size=TRAIN_RATIO,
    stratify=labels,
    random_state=42,
)

val_imgs, test_imgs, val_labels, test_labels, val_paths, test_paths = train_test_split(
    temp_imgs,
    temp_labels,
    temp_paths,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42,
)

# ==========================================================
# COPY FUNCTION
# ==========================================================

def copy_split(imgs, labels, rel_paths, split_name):

    counter = defaultdict(int)

    for img, label, rel in zip(imgs, labels, rel_paths):

        destination = DESTINATION / split_name / label / rel

        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, destination)

        counter[label] += 1

    return counter

# ==========================================================
# COPY FILES
# ==========================================================

train_count = copy_split(train_imgs, train_labels, train_paths, "train")
val_count = copy_split(val_imgs, val_labels, val_paths, "val")
test_count = copy_split(test_imgs, test_labels, test_paths, "test")

# ==========================================================
# REPORT
# ==========================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for folder, disease in CLASS_MAPPING.items():

    print(f"\n{disease}")

    print(f"Train : {train_count[folder]}")

    print(f"Val   : {val_count[folder]}")

    print(f"Test  : {test_count[folder]}")

print("\n")
print("=" * 70)
print("TOTAL")
print("=" * 70)

print(f"Train : {len(train_imgs)}")
print(f"Val   : {len(val_imgs)}")
print(f"Test  : {len(test_imgs)}")

print("\nDataset prepared successfully.")
print("=" * 70)