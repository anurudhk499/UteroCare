import timm

import torch.nn as nn

# ─────────────────────────────
# MODEL FUNCTION
# ─────────────────────────────

def build_model(num_classes):

    model = timm.create_model(

        "tf_efficientnetv2_s",

        pretrained=True
    )
    print(model)

    # ─────────────────────────
    # GET FEATURE SIZE
    # ─────────────────────────

    in_features = model.classifier.in_features

    # ─────────────────────────
    # CUSTOM CLASSIFIER
    # ─────────────────────────

    model.classifier = nn.Sequential(

        nn.Linear(in_features,384),

        nn.BatchNorm1d(384),

        nn.SiLU(),

        nn.Dropout(0.35),

        nn.Linear(384,128),

        nn.BatchNorm1d(128),

        nn.SiLU(),

        nn.Dropout(0.25),

        nn.Linear(128,num_classes)
    )

    return model