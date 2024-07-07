from django.urls import path
from . import views

urlpatterns = [
    path("", views.main, name="main"),
    path("upload_html/", views.upload_html, name="upload_html"),
    path("ask_question_api/", views.ask_question_api, name="ask_question_api"),
    path("ask_question_model/", views.ask_question_model, name="ask_question_model"),
]
