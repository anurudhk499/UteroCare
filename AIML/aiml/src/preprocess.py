import os
import cv2
import shutil
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================

INPUT_DIR = "../processed_dataset_v2"
OUTPUT_DIR = "../preprocessed_dataset_v2"

IMAGE_SIZE = 224

# ==========================================
# FUNCTIONS
# ==========================================

def resize_image(image):
    return cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )


def normalize_image(image):
    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        3,
        3,
        7,
        21
    )


# ==========================================
# COUNTERS
# ==========================================

original_count = 0
denoise_count = 0

# ==========================================
# MAIN LOOP
# ==========================================

for split in ["train", "val", "test"]:

    print(f"\n========== {split.upper()} ==========")

    split_input = os.path.join(INPUT_DIR, split)

    for cls in os.listdir(split_input):

        input_class = os.path.join(split_input, cls)

        output_class = os.path.join(
            OUTPUT_DIR,
            split,
            cls
        )

        os.makedirs(
            output_class,
            exist_ok=True
        )

        images = os.listdir(input_class)

        print(f"\nProcessing {cls}")

        for idx, image_name in enumerate(tqdm(images)):

            img_path = os.path.join(
                input_class,
                image_name
            )

            image = cv2.imread(img_path)

            if image is None:
                print(f"Skipped: {img_path}")
                continue

            image = resize_image(image)
            image = normalize_image(image)

            # ---------------------------------
            # Save ORIGINAL (all splits)
            # ---------------------------------

            original_path = os.path.join(
                output_class,
                f"orig_{idx}_{image_name}"
            )

            cv2.imwrite(
                original_path,
                image
            )

            original_count += 1

            # ---------------------------------
            # TRAIN ONLY
            # Save DENOISED VERSION
            # ---------------------------------

            if split == "train":

                denoise = denoise_image(image)

                denoise_path = os.path.join(
                    output_class,
                    f"denoise_{idx}_{image_name}"
                )

                cv2.imwrite(
                    denoise_path,
                    denoise
                )

                denoise_count += 1

# ==========================================
# SUMMARY
# ==========================================

print("\n===================================")
print(" PREPROCESSING COMPLETED")
print("===================================")

print(f"\nOriginal Images : {original_count}")
print(f"Denoised Images : {denoise_count}")
print(f"Total Images    : {original_count + denoise_count}")