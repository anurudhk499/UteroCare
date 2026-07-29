
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from PIL import Image
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torchvision.transforms as transforms
import torch
from .model_loader import model, classes, device
import cv2
from .models import Prediction
import torchvision.transforms as transforms
from django.shortcuts import render
from .gradcam_utils import generate_gradcam
from .risk_model_loader import predict_risk
from .symptom_model_loader import predict_symptom
from .fusion_engine import fuse_predictions
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .chatbot.rag import generate_answer


@csrf_exempt
def chatbot(request):

    if request.method != "POST":
        return JsonResponse({"error": "POST request required"})

    try:
        data = json.loads(request.body)

        question = data.get("question", "").strip()
        patient = data.get("patient", {})
        patient_context = f"""
Age : {patient.get("age","N/A")}

BMI : {patient.get("bmi","N/A")}

Pain Score : {patient.get("pain","N/A")}/10

MRI Prediction : {patient.get("mri_prediction","N/A")}

Symptom Prediction : {patient.get("symptom_prediction","N/A")}

Final Prediction : {patient.get("final_prediction","N/A")}

Confidence : {patient.get("confidence","N/A")}%

Risk : {patient.get("risk","N/A")}

Symptoms :

{patient.get("symptoms","None")}
"""

        if not question:
            return JsonResponse({"error": "Question is required"})

        answer = generate_answer(question, patient_context)

        return JsonResponse({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })
# REGISTER

def register(request):

    if request.method == "POST":

        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:

            messages.error(request, "Passwords do not match")
            return redirect('register')

        if User.objects.filter(username=username).exists():

            messages.error(request, "Username already exists")
            return redirect('register')

        if User.objects.filter(email=email).exists():

            messages.error(request, "Email already exists")
            return redirect('register')

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        user.save()

        messages.success(
            request,
            "Registration successful. Please login."
        )

        return redirect('login')

    return render(request, 'register.html')


# LOGIN

def login_view(request):

    if request.method == "POST":

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(request, user)

            messages.success(request, "Login successful")

            return redirect('dashboard')

        else:

            messages.error(request, "Invalid username or password")

            return redirect('login')

    return render(request, 'login.html')


# LOGOUT

def logout_view(request):

    logout(request)

    messages.success(request, "Logged out successfully")

    return redirect('login')

@login_required(login_url='login')

def dashboard(request):

    predictions = Prediction.objects.filter(

        user=request.user

    ).order_by('-analyzed_at')

    total_predictions = predictions.count()

    latest_prediction = predictions.first()

    latest_risk = 0

    latest_disease = "Awaiting Analysis"

    if latest_prediction:

        latest_risk = (
            latest_prediction.fusion_risk_score
        )

        latest_disease = (
            latest_prediction.disease
        )

    health_score = max(

        100 - int(latest_risk),

        0
    )

    return render(

        request,

        'dashboard.html',

        {

            "predictions": predictions,

            "total_predictions":
                total_predictions,

            "latest_risk":
                latest_risk,

            "latest_disease":
                latest_disease,

            "health_score":
                health_score
        }
    )

@login_required(login_url='login')
def doctor_dashboard(request):

    return render(request, 'doctor-dashboard.html')





transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])





@csrf_exempt
def predict_disease(request):

    if request.method == "POST":

        try:

            # =====================
            # GET IMAGE
            # =====================

            image = request.FILES.get("mri")

            if not image:

                return JsonResponse({

                    "status": "error",

                    "message": "MRI image not uploaded"

                })

            # =====================
            # PROCESS IMAGE
            # =====================

            img = Image.open(image).convert("RGB")

            original_image = img.copy()

            img = transform(img)

            img = img.unsqueeze(0)

            img = img.to(device)

            # =====================
            # MODEL PREDICTION
            # =====================

            img.requires_grad_()

            with torch.enable_grad():

                output = model(img)

                probs = torch.softmax(output, dim=1)

                confidence, pred = torch.max(probs, 1)

            # =====================
            # GRADCAM
            # =====================
            original_save_path = os.path.join(
                "static",
                "outputs",
                "original_scan.png"
            )

            original_image.save(
                original_save_path
            )
            gradcam_path, affected_area = generate_gradcam(

                model,

                img,

                original_image,

                pred.item()

            )
            

            # =====================
            # DISEASE LABEL
            # =====================

  
            disease = classes[pred.item()]
            display_names = {

                "adenomyosis": "Adenomyosis",

                "cancer": "Endometrial Cancer",

                "fibroid": "Fibroid",

                "normal": "Normal Uterus"

            }

            disease = display_names.get(

                disease,

                disease

            )
            mri_prediction = disease

           

            if disease == "Normal Uterus":

                            gradcam_path = (
                                "/static/outputs/original_scan.png"
                            )

                            affected_area = 0
            confidence = round(

                confidence.item() * 100,

                2

            )
            mri_confidence = confidence
            all_probs = probs.detach().cpu().numpy()[0]

            # =====================
            # CLINICAL INPUTS
            # =====================

            age = float(

                request.POST.get(

                    "age",

                    30

                )

            )

            height = float(

                request.POST.get(

                    "height",

                    160

                )

            )

            weight = float(

                request.POST.get(

                    "weight",

                    60

                )

            )

            bmi = weight / (

                (height / 100) ** 2

            )

           

            pain_score = float(

                request.POST.get(

                    "pain_score",

                    5

                )

            )
            menopause = "Yes" if request.POST.get("menopause_status") == "1" else "No"
            heavy_bleeding = request.POST.get("heavy_bleeding_level", "None")
            menstrual_cramps = request.POST.get("menstrual_cramps_level", "None")
            pelvic_pain = request.POST.get("pelvic_pain_level", "None")
            patient = {

    "Age": int(age),

    "Menopause": menopause,

    "Heavy_Menstrual_Bleeding": heavy_bleeding,

    "Menstrual_Cramps": menstrual_cramps,

    "Pelvic_Pain": pelvic_pain,

    "Bleeding_Between_Periods":
        "Yes" if request.POST.get("bleeding_between_periods") == "1" else "No",

    "Bleeding_After_Menopause":
        "Yes" if request.POST.get("bleeding_after_menopause") == "1" else "No",

    "Periods_Longer_Than_7_Days":
        "Yes" if request.POST.get("periods_longer_than_7_days") == "1" else "No",

    "Pain_During_Intercourse":
        "Yes" if request.POST.get("pain_during_intercourse") == "1" else "No",

    "Frequent_Urination":
        "Yes" if request.POST.get("frequent_urination") == "1" else "No",

    "Constipation":
        "Yes" if request.POST.get("constipation") == "1" else "No",

    "Pelvic_Pressure_or_Fullness":
        "Yes" if request.POST.get("pelvic_pressure") == "1" else "No",

    "Abdominal_Swelling":
        "Yes" if request.POST.get("abdominal_swelling") == "1" else "No",

    "Lower_Back_Pain":
        "Yes" if request.POST.get("lower_back_pain") == "1" else "No",

    "Abnormal_Vaginal_Discharge":
        "Yes" if request.POST.get("abnormal_discharge") == "1" else "No",

    "Fatigue":
        "Yes" if request.POST.get("fatigue") == "1" else "No",

    "Diagnosed_Anemia":
        "Yes" if request.POST.get("diagnosed_anemia") == "1" else "No",

    "Difficulty_Conceiving":
        "Yes" if request.POST.get("difficulty_conceiving") == "1" else "No",

    "BMI": bmi,

    "Pain_Level": pain_score,

    "MRI_Confidence": mri_confidence,

    "Affected_Area_Pct": affected_area,

    "Final_Diagnosis": mri_prediction

}

      
            # =====================
            # MRI PROBABILITIES
            # =====================
            adeno_prob = float(all_probs[0])
            cancer_prob = float(all_probs[1])
            fibroid_prob = float(all_probs[2])
            normal_prob = float(all_probs[3])
          
            
            

            mri_probs = {

                "Adenomyosis": adeno_prob,

                "Endometrial Cancer": cancer_prob,

                "Fibroid": fibroid_prob,

                "Normal Uterus": normal_prob

            }
            symptom_result = predict_symptom(patient)

            symptom_prediction = symptom_result["prediction"]

            symptom_confidence = symptom_result["confidence"]

            fusion_result = fuse_predictions(
                mri_prediction,
                symptom_prediction,
                mri_probs
            )

            final_prediction = fusion_result["prediction"]
            if final_prediction == "Endometrial Cancer":
                final_prediction = "Endometrial_Cancer"

            elif final_prediction == "Normal Uterus":
                final_prediction = "Normal"

            final_confidence = fusion_result["confidence"]

            fusion_mode = fusion_result["mode"]

            patient["Final_Diagnosis"] = final_prediction

            fusion_risk_score = predict_risk(patient)
            fusion_risk_score = round(float(fusion_risk_score), 2)
            fusion_result = fuse_predictions(
                mri_prediction,
                symptom_prediction,
                mri_probs
            )


            fusion_risk_score = predict_risk(patient)
                # =====================
                # RISK LEVEL
                # =====================

            if fusion_risk_score >= 80:
                risk_level = "very_high"

            elif fusion_risk_score >= 51:
                risk_level = "high"

            elif fusion_risk_score >= 26:
                risk_level = "moderate"

            else:
                risk_level = "low"
            print("========== FUSION ==========")
            print("MRI Prediction      :", mri_prediction)
            print("Symptom Prediction :", symptom_prediction)
            print("Final Prediction   :", final_prediction)
            print("Fusion Mode        :", fusion_mode)
            print("============================")
                        # =====================
            # RESPONSE
            # =====================
            Prediction.objects.create(

                user=request.user,

                disease=final_prediction,

                confidence=final_confidence,

                fusion_risk_score=fusion_risk_score,

                gradcam_image=gradcam_path,

                affected_area=str(affected_area),
                mri_prediction=mri_prediction,

                symptom_prediction=symptom_prediction,

                fusion_mode=fusion_mode,

                risk_level=risk_level
            )
            if risk_level == "low":
                recommendation = "Routine follow-up recommended."

            elif risk_level == "moderate":
                recommendation = "Consult a gynecologist for further evaluation."

            elif risk_level == "high":
                recommendation = "Medical consultation is recommended as soon as possible."

            else:
                recommendation = "Immediate specialist consultation is strongly recommended."
            return JsonResponse({
                "recommendation": recommendation,
                "bmi": round(bmi,1),
                "pain_level": pain_score,
                "affected_area": affected_area,
                
                "status": "success",
                "disease": final_prediction,

                "mri_prediction": mri_prediction,

                "symptom_prediction": symptom_prediction,

                "fusion_mode": fusion_mode,

                "confidence": final_confidence,

                "fusion_risk_score":

                    fusion_risk_score,

                "risk_level":

                    risk_level,

                "description":

                    f"AI detected possible {final_prediction}.",

                "mode": "fusion",

                            "adeno_risk":
                    f"{round(adeno_prob * 100, 1)}%",

                "endo_risk":
                    f"{round(cancer_prob * 100, 1)}%",

                "fibroid_risk":
                    f"{round(fibroid_prob * 100, 1)}%",

                "normal_score":
                    f"{round(normal_prob * 100, 1)}%",

                                "next_scan":

                    "6 Months",

                "gradcam":

                    gradcam_path,

               

                "analyzed_at":

                    datetime.now().strftime(

                        "%d/%m/%Y, %I:%M %p"

                    )

            })

        except Exception as e:

            return JsonResponse({

                "status": "error",

                "message": str(e)

            })

    return JsonResponse({

        "status": "error",

        "message": "Invalid request"

    })



@login_required(login_url='login')
def predictions(request):

    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by('-analyzed_at')

    return render(
        request,
        'predictions.html',
        {
            'predictions': predictions
        }
    )


@login_required(login_url='login')
def reports(request):

    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by('-analyzed_at')

    return render(
        request,
        'reports.html',
        {
            'predictions': predictions
        }
    )


@login_required(login_url='login')
def appointments(request):

    return render(
        request,
        'appointments.html'
    )


@login_required(login_url='login')
def analytics(request):

    predictions = Prediction.objects.filter(
        user=request.user
    )

    return render(
        request,
        'analytics.html',
        {
            'predictions': predictions
        }
    )


@login_required(login_url='login')
def symptom_tracker(request):

    return render(
        request,
        'symptom_tracker.html'
    )


@login_required(login_url='login')
def medications(request):

    return render(
        request,
        'medications.html'
    )


@login_required(login_url='login')
def my_files(request):

    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by('-analyzed_at')

    return render(
        request,
        'my_files.html',
        {
            'predictions': predictions
        }
    )
    