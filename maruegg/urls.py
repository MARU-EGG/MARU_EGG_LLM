from django.urls import path
from . import views

urlpatterns = [
    # main
    path("", views.main, name="main"),
    path("upload_html/", views.upload_html, name="upload_html"),
    path("upload_pdf/", views.upload_pdf, name="upload_pdf"),
    path("ask_question_api/", views.ask_question_api, name="ask_question_api"),
    path('pdf_link/<int:page>/', views.pdf_link, name='pdf_link'),
    path('pdf_direct_link/<int:page>/', views.pdf_direct_link, name='pdf_direct_link'),
    # delete APIs
    path("delete_documents_all/", views.delete_documents_all, name="delete_documents_all"),
    path("delete_documents_by_type/", views.delete_documents_by_type, name="delete_documents_by_type"),
    path("delete_documents_by_type_and_category/", views.delete_documents_by_type_and_category, name="delete_documents_by_type_and_category"),
    # retrieve APIs
    path("retrieve_documents_all/", views.retrieve_documents_all, name="retrieve_documents_all"),
    path("retrieve_documents_by_type/", views.retrieve_documents_by_type, name="retrieve_documents_by_type"),
    path("retrieve_documents_by_type_and_category/", views.retrieve_documents_by_type_and_category, name="retrieve_documents_by_type_and_category"),
    # test
    path("upload_html_show/", views.upload_html_show, name="upload_html_show"),
]