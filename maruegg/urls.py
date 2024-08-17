from django.urls import path
from .views import main, upload_pdf, ask_question_api, delete_documents, retrieve_documents

urlpatterns = [
    # main
    path("", main, name="main"),
    path("upload_pdf/", upload_pdf, name="upload_pdf"),
    path("ask_question_api/", ask_question_api, name="ask_question_api"),
    # delete APIs
    path("delete_documents/", delete_documents, name="delete_documents"),
    # retrieve APIs
    path("retrieve_documents/", retrieve_documents, name="retrieve_documents"),
]