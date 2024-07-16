from django.urls import path
from . import views

urlpatterns = [
    path("", views.main, name="main"),
    path("upload_html/", views.upload_html, name="upload_html"),
    path("ask_question_api/", views.ask_question_api, name="ask_question_api"),
    # path("ask_question_model/", views.ask_question_model, name="ask_question_model"),
    # delete APIs
    path("delete_documents_all/", views.delete_documents_all, name="delete_documents_all"),
    path("delete_documents_by_type/", views.delete_documents_by_type, name="delete_documents_by_type"),
    path("delete_documents_by_type_and_category/", views.delete_documents_by_type_and_category, name="delete_documents_by_type_and_category"),
    # retrieve APIs
    path("retrieve_documents_all/", views.retrieve_documents_all, name="retrieve_documents_all"),
    path("retrieve_documents_by_type/", views.retrieve_documents_by_type, name="retrieve_documents_by_type"),
    path("retrieve_documents_by_type_and_category/", views.retrieve_documents_by_type_and_category, name="retrieve_documents_by_type_and_category"),
]