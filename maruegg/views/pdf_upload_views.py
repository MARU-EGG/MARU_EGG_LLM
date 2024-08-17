import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from ..models import Document1, Document2, Document3
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
MEDIA_DOCUMENT_URL = os.path.join(settings.MEDIA_ROOT, 'documents')
MEDIA_FILES_URL = os.path.join(settings.MEDIA_ROOT, 'files')

def main(request):
    return render(request, "maruegg/upload_html_file.html")

@swagger_auto_schema(
    method='post',
    operation_description="LLM모델이 답변할 기반이 되는 pdf 파일 업로드 api입니다. type, category에 따라 db에 저장되는 위치가 달라집니다.",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_FORM,
            description='Type of the document (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_FORM,
            description='Category of the document (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
        openapi.Parameter(
            'pdf_file',
            openapi.IN_FORM,
            description='PDF file to be uploaded',
            required=True,
            type=openapi.TYPE_FILE
        ),
    ],
    responses={
        200: openapi.Response(description="Success"),
        400: openapi.Response(description="Invalid request"),
    }
)
@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_pdf(request):
    if request.method == "POST" and 'pdf_file' in request.FILES:
        pdf_file = request.FILES["pdf_file"]
        doc_type = request.POST.get("type")
        doc_category = request.POST.get("category")
        if doc_type not in ["수시", "정시", "편입학"]:
            return JsonResponse({"error": "Invalid type provided"}, status=400)
        if doc_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
            return JsonResponse({"error": "Invalid category provided"}, status=400)

        model_class = None
        if doc_type == "수시":
            model_class = Document1
        elif doc_type == "정시":
            model_class = Document2
        elif doc_type == "편입학":
            model_class = Document3

        logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
        model_class.objects.filter(category=doc_category).delete()

        filename = f"{doc_type}_{doc_category}.pdf"
        save_file(pdf_file, filename)

        logger.debug(f"PDF file {filename} saved to documents and files folder.")
        return JsonResponse({"message": "File uploaded successfully"}, status=200)
    return JsonResponse({"error": "Invalid request"}, status=400)

def save_file(file, filename):
    documents_path = os.path.join(MEDIA_DOCUMENT_URL, filename)
    if os.path.exists(documents_path):
        os.remove(documents_path)
    with open(documents_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    files_path = os.path.join(MEDIA_FILES_URL, filename)
    if os.path.exists(files_path):
        os.remove(files_path)
    with open(files_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return documents_path, files_path