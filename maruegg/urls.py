from django.urls import path
from . import views

urlpatterns = [
    path("", views.main, name="main"),
    path("upload/", views.upload_pdf, name="upload_pdf"),
    path("upload_html/", views.upload_html, name="upload_html"),
]
