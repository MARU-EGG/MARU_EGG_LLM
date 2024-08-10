from django.urls import path
from . import views

urlpatterns = [
    # main
    path("", views.main, name="main"),
    path("upload_pdf/", views.upload_pdf, name="upload_pdf"),
    path("ask_question_api/", views.ask_question_api, name="ask_question_api"),
    # delete APIs
    path("delete_documents/", views.delete_documents, name="delete_documents"),
    # retrieve APIs
    path("retrieve_documents/", views.retrieve_documents, name="retrieve_documents"),
    # test
    path("upload_html_show/", views.upload_html_show, name="upload_html_show"),
]