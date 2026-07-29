import timm
import torch.nn as nn

from configs.config import MODEL_NAME, NUM_CLASSES, DROPOUT


def build_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model

def freeze_backbone(model):

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
def unfreeze_model(model):

    for param in model.parameters():
        param.requires_grad = True

    return model