
import torch
import torch.nn as nn

class FusionRiskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1286, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 1),

            nn.Sigmoid()
        )

    def forward(

        self,
        mri_features,
        clinical_features
    ):

        x = torch.cat(

            [
                mri_features,
                clinical_features
            ],

            dim=1
        )

        return self.network(x)

