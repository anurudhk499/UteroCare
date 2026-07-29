from django.urls import path
from django.urls import path, include
from . import views
from .views import dashboard, register, login_view, logout_view
from django.contrib.auth import views as auth_views
from .views import predict_disease
urlpatterns = [

    # AUTH
    path('register/', register, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),

    # PASSWORD RESET
    path(
        'password-reset/',
        auth_views.PasswordResetView.as_view(
            template_name='forgot_password.html'
        ),
        name='password_reset'
    ),

    path(
        'password-reset/done/',
        auth_views.PasswordResetDoneView.as_view(
            template_name='password_reset_done.html'
        ),
        name='password_reset_done'
    ),

    path(
        'reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(
            template_name='password_reset_confirm.html'
        ),
        name='password_reset_confirm'
    ),

    path(
        'reset-complete/',
        auth_views.PasswordResetCompleteView.as_view(
            template_name='password_reset_complete.html'
        ),
        name='password_reset_complete'
    ),

    # MAIN
    path('', views.dashboard, name='dashboard'),

    path('dashboard/', views.dashboard, name='dashboard'),

    path(
        'doctor-dashboard/',
        views.doctor_dashboard,
        name='doctor-dashboard'
    ),

    path(
        'predict/',
        predict_disease,
        name='predict_disease'
    ),

    # SIDEBAR PAGES
    path('predictions/', views.predictions, name='predictions'),
    path('reports/', views.reports, name='reports'),
    path('appointments/', views.appointments, name='appointments'),
    path('analytics/', views.analytics, name='analytics'),
    path('symptom_tracker/', views.symptom_tracker, name='symptom_tracker'),
    path('medications/', views.medications, name='medications'),
    path('my_files/', views.my_files, name='my_files'),
    path("chatbot/", views.chatbot, name="chatbot"),
    path("accounts/", include("allauth.urls")),
    

]


