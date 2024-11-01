from django.urls import path
from .views import main, upload_pdf, ask_question_api, delete_documents, retrieve_documents, create_prompt, get_prompt, update_prompt, delete_prompt


urlpatterns = [
    # main
    path("", main, name="main"),
    path("upload_pdf/", upload_pdf, name="upload_pdf"),
    path("ask_question_api/", ask_question_api, name="ask_question_api"),
    # delete APIs
    path("delete_documents/", delete_documents, name="delete_documents"),
    # retrieve APIs
    path("retrieve_documents/", retrieve_documents, name="retrieve_documents"),
    # prompt APIs
    path('create-prompt/', create_prompt, name='create_prompt'),
    path('get-prompt/', get_prompt, name='get_prompt'),
    path('update-prompt/', update_prompt, name='update_prompt'),
    path('delete-prompt/', delete_prompt, name='delete_prompt'),
]