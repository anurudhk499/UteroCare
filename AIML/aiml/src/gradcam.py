import torch
import cv2
import numpy as np

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import build_model
from config import *

# -----------------------------
# LOAD MODEL
# -----------------------------

model = build_model(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        weights_only=True
    )
)

model.to(DEVICE)

model.eval()

# -----------------------------
# TARGET LAYER
# -----------------------------

target_layers = [model.conv_head]

cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)

# -----------------------------
# IMAGE PATH
# -----------------------------

image_path = "../test_image.jpg"

# -----------------------------
# LOAD IMAGE
# -----------------------------

image = Image.open(
    image_path
).convert("RGB")

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

input_tensor = transform(
    image
).unsqueeze(0).to(DEVICE)

# -----------------------------
# MODEL PREDICTION
# -----------------------------

with torch.no_grad():

    output = model(input_tensor)

    probs = torch.softmax(
        output,
        dim=1
    )[0]

    pred_class = torch.argmax(
        probs
    ).item()

    confidence = probs[
        pred_class
    ].item()

# -----------------------------
# GENERATE CAM
# -----------------------------

grayscale_cam = cam(
    input_tensor=input_tensor
)[0]

# -----------------------------
# ORIGINAL IMAGE
# -----------------------------

rgb_img = np.array(
    image.resize((224,224))
).astype(np.float32) / 255.0

# -----------------------------
# CREATE HEATMAP OVERLAY
# -----------------------------

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# -----------------------------
# THRESHOLD HEATMAP
# -----------------------------

cam_uint8 = np.uint8(
    grayscale_cam * 255
)

_, thresh = cv2.threshold(
    cam_uint8,
    150,
    255,
    cv2.THRESH_BINARY
)

# -----------------------------
# FIND CONTOURS
# -----------------------------

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# DRAW LARGEST CONTOUR
# -----------------------------

if len(contours) > 0:

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # AREA

    lesion_area = cv2.contourArea(
        largest_contour
    )

    total_area = 224 * 224

    affected_percent = (
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

else:

    affected_percent = 0

# -----------------------------
# SAVE OUTPUT
# -----------------------------

output_path = (
    "../outputs/region_result.png"
)

cv2.imwrite(

    output_path,

    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

# -----------------------------
# RESULTS
# -----------------------------

print("\nAI REGION ANALYSIS\n")

print(
    f"Prediction: {CLASS_NAMES[pred_class]}"
)

print(
    f"Confidence: {confidence:.4f}"
)

print(
    f"Approx Affected Area: "
    f"{affected_percent:.2f}%"
)

print(
    f"Saved Result: {output_path}"
)