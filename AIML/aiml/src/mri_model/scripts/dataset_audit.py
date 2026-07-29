import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = Path("../uterus_dataset")
OUTPUT_PATH = Path("../outputs")

OUTPUT_PATH.mkdir(exist_ok=True)

# ==========================================================
# CLASS MAPPING
# ==========================================================

CLASS_MAPPING = {
    "adeno_mri": "Adenomyosis",
    "fibroid_mri": "Fibroid",
    "endometrial_cancer_mri": "Endometrial_Cancer",
    "normal_uterus_MRI": "Normal"
}

# ==========================================================
# VARIABLES
# ==========================================================

metadata = []

duplicates = []
corrupted = []

hash_dict = {}

class_counter = Counter()
thickness_counter = Counter()
resolution_counter = Counter()
format_counter = Counter()

print("=" * 75)
print("UTEROCARE MRI DATASET AUDIT")
print("=" * 75)

# ==========================================================
# FUNCTION
# ==========================================================

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def process_image(image_path, disease, thickness):

    global metadata

    try:

        img = Image.open(image_path)

        width, height = img.size

        fmt = img.format

        resolution_counter[f"{width}x{height}"] += 1
        format_counter[fmt] += 1

        file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()

        if file_hash in hash_dict:
            duplicates.append(str(image_path))
        else:
            hash_dict[file_hash] = str(image_path)

        metadata.append({
            "Disease": disease,
            "Thickness": thickness,
            "Image": image_path.name,
            "Width": width,
            "Height": height,
            "Format": fmt,
            "Path": str(image_path)
        })

        class_counter[disease] += 1

    except Exception:

        corrupted.append(str(image_path))


# ==========================================================
# MAIN LOOP
# ==========================================================

for folder_name, disease in CLASS_MAPPING.items():

    disease_folder = DATASET_PATH / folder_name

    if not disease_folder.exists():
        print(f"\nMissing Folder : {folder_name}")
        continue

    print(f"\nScanning {disease}...")

    # --------------------------------------------------
    # CASE 1 : Images directly inside folder
    # --------------------------------------------------

    direct_images = [
        f for f in disease_folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if len(direct_images) > 0:

        thickness_counter[(disease, "Normal")] = len(direct_images)

        for img in direct_images:
            process_image(img, disease, "Normal")

        continue

    # --------------------------------------------------
    # CASE 2 : Thickness folders
    # --------------------------------------------------

    for subfolder in disease_folder.iterdir():

        if not subfolder.is_dir():
            continue

        thickness = subfolder.name

        count = 0

        for img in subfolder.iterdir():

            if img.suffix.lower() not in VALID_EXTENSIONS:
                continue

            process_image(img, disease, thickness)

            count += 1

        thickness_counter[(disease, thickness)] = count

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(metadata)

df.to_csv(OUTPUT_PATH / "metadata.csv", index=False)

# ==========================================================
# REPORT
# ==========================================================

print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

for k, v in class_counter.items():
    print(f"{k:25s}: {v}")

print("\n")
print("=" * 75)
print("THICKNESS DISTRIBUTION")
print("=" * 75)

for (disease, thickness), count in sorted(thickness_counter.items()):
    print(f"{disease:22s} | {thickness:30s}: {count}")

print("\n")
print("=" * 75)
print("IMAGE RESOLUTIONS")
print("=" * 75)

for res, count in resolution_counter.items():
    print(f"{res:15s}: {count}")

print("\n")
print("=" * 75)
print("IMAGE FORMATS")
print("=" * 75)

for fmt, count in format_counter.items():
    print(f"{fmt:10s}: {count}")

print("\n")
print("=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"Total Images      : {len(metadata)}")
print(f"Duplicate Images  : {len(duplicates)}")
print(f"Corrupted Images  : {len(corrupted)}")

print(f"\nMetadata Saved : {OUTPUT_PATH / 'metadata.csv'}")

if duplicates:

    with open(OUTPUT_PATH / "duplicates.txt", "w") as f:
        for d in duplicates:
            f.write(d + "\n")

    print("Duplicate list saved.")

if corrupted:

    with open(OUTPUT_PATH / "corrupted.txt", "w") as f:
        for c in corrupted:
            f.write(c + "\n")

    print("Corrupted list saved.")

print("\nDataset Audit Completed Successfully!")
print("=" * 75)