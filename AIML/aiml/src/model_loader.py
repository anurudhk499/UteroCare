import torch
import timm

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# CLASS NAMES
classes = [

    "adenomyosis",
    "endometrial_cancer",
    "fibroid",
    "normal_uterus"

]

# CREATE MODEL
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=4
)

# LOAD WEIGHTS
model.load_state_dict(
    torch.load(
        "models/baseline_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()