
import os
from turtle import width
import cv2
from sympy import python
import torch
import numpy as np

from PIL import Image

from django.conf import settings


from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    model,
    input_tensor,
    original_image,
    predicted_class
):

    # LAST LAYER
    target_layers = [model.bn2]



    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    # ORIGINAL IMAGE
    rgb_img = np.array(
        original_image.resize((224,224))
    ).astype(np.float32) / 255.0

    # HEATMAP
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # THRESHOLD
    cam_uint8 = np.uint8(
        grayscale_cam * 255
    )

    _, thresh = cv2.threshold(
        cam_uint8,
        150,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    attention_percent = 0

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        lesion_area = cv2.contourArea(
            largest_contour
        )

        height, width = rgb_img.shape[:2]

        total_area = height * width

            # ==========================================
        # AI HIGHLIGHTED REGION
        # ==========================================

        height, width = rgb_img.shape[:2]

        total_area = height * width

        attention_percent = (
            lesion_area / total_area
        ) * 100

        # DRAW CONTOUR
        cv2.drawContours(
            visualization,
            [largest_contour],
            -1,
            (0,255,0),
            2
        )

        # BOUNDING BOX
        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        cv2.rectangle(
            visualization,
            (x,y),
            (x+w, y+h),
            (255,0,0),
            2
        )

        # LABEL
    cv2.putText(
        visualization,
        f"Area: {attention_percent:.1f}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # SAVE

    output_dir = os.path.join(
    settings.BASE_DIR,
    "static",
    "outputs"
    )

    os.makedirs(
    output_dir,
    exist_ok=True
)

    output_path = os.path.join(
    output_dir,
    "gradcam_result.png"
)




    cv2.imwrite(
    output_path,
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)


    return (
    "/static/outputs/gradcam_result.png",
    round(attention_percent, 2)
    )


