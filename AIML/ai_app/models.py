from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    disease = models.CharField(
        max_length=120
    )

    confidence = models.FloatField()

    fusion_risk_score = models.FloatField()

    gradcam_image = models.TextField()

    affected_area = models.CharField(
        max_length=50
    )

    analyzed_at = models.DateTimeField(
        auto_now_add=True
    )
    mri_prediction = models.CharField(max_length=100, blank=True, null=True)

    symptom_prediction = models.CharField(max_length=100, blank=True, null=True)

    fusion_prediction = models.CharField(max_length=100, blank=True, null=True)

    fusion_mode = models.CharField(max_length=50, blank=True, null=True)

    risk_score = models.FloatField(default=0)

    risk_level = models.CharField(max_length=30, blank=True, null=True)
    def __str__(self):

        return (
            f"{self.user.username} - "
            f"{self.disease}"
        )