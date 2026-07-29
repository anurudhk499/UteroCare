import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "../dataset"

OUTPUT_DIR = "../processed_dataset_v2"

classes = [

    "adenomyosis",

    "cancer",

    "fibroid",

    "normal"
]
for cls in classes:

    class_path = os.path.join(
        SOURCE_DIR,
        cls
    )

    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)
    print(f"Processing class '{cls}' with {total} images.")
    train_end = int(total * 0.7)

    val_end = int(total * 0.85)

    train_images = images[:train_end]

    val_images = images[
        train_end:val_end
    ]

    test_images = images[val_end:]

    split_data = {

        "train": train_images,

        "val": val_images,

        "test": test_images
    }

    for split, split_images in split_data.items():

        split_folder = os.path.join(

            OUTPUT_DIR,

            split,

            cls
        )

        os.makedirs(
            split_folder,
            exist_ok=True
        )

        for image_name in split_images:

            src = os.path.join(
                class_path,
                image_name
            )

            dst = os.path.join(
                split_folder,
                image_name
            )

            shutil.copy(src, dst)

print(
    "\nDataset splitting completed successfully."
)