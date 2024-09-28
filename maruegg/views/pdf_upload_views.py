import os
import re
import shutil
import pymupdf4llm
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
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

category_mapping = {
    "모집요강": "admission_guideline",
    "입시결과": "admission_results",
    "기출문제": "past_exams",
    "대학생활": "campus_life",
    "면접/실기": "interview_practice"
}

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

        if not validate_document_type_and_category(doc_type, doc_category):
            return JsonResponse({"error": "Invalid type or category provided"}, status=400)

        model_class = get_model_class(doc_type)

        delete_existing_documents(model_class, doc_category)
        delete_vector_db_folder(doc_type, doc_category)

        filename = f"{doc_type}_{doc_category}.pdf"
        file_path = save_file(pdf_file, filename)

        if doc_category == "입시결과":
            parse_pdf_file_basic(file_path, doc_type, doc_category)
        else:
            parse_pdf_file(file_path, doc_type, doc_category)

        logger.debug(f"PDF file {filename} processed and saved to the database.")
        return JsonResponse({"message": "File uploaded and processed successfully"}, status=200)

    return JsonResponse({"error": "Invalid request"}, status=400)

def validate_document_type_and_category(doc_type, doc_category):
    return doc_type in ["수시", "정시", "편입학"] and doc_category in category_mapping.keys()

def get_model_class(doc_type):
    if doc_type == "수시":
        return Document1
    elif doc_type == "정시":
        return Document2
    elif doc_type == "편입학":
        return Document3
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

def delete_existing_documents(model_class, doc_category):
    logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
    model_class.objects.filter(category=doc_category).delete()

def delete_vector_db_folder(doc_type, doc_category):
    model_class = get_model_class(doc_type)
    class_name = model_class.__name__
    english_category = category_mapping.get(doc_category, doc_category)

    persist_directory = os.path.join(settings.BASE_DIR, f"vectorDB/{class_name}/{english_category}_vectorDB")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        logger.debug(f"Deleted vector store at {persist_directory}")
    else:
        logger.debug(f"Vector store path {persist_directory} does not exist, no need to delete.")

def save_file(file, filename):
    documents_path = os.path.join(MEDIA_DOCUMENT_URL, filename)
    if os.path.exists(documents_path):
        os.remove(documents_path)

    with open(documents_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    return documents_path

def parse_pdf_file(file_path, doc_type, doc_category):
    """
    모집요강 파일을 파싱하고 처리하는 함수.
    """
    pdf_document = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    print("ㄴㅇㄹㄴㅇㄹ:" + pymupdf4llm.__version__)

    # pageGap 설정 (수시: 1, 정시: 2)
    page_gap = 1 if doc_type == "수시" else 2

    toc_mapping = extract_toc(pdf_document, page_gap)
    add_titles_to_pages(pdf_document, toc_mapping)

    modify_text_based_on_metadata(pdf_document)

    title = os.path.basename(file_path)
    save_parsed_pdf_to_db(pdf_document, doc_type, doc_category, title)

def parse_pdf_file_basic(file_path, doc_type, doc_category):
    """
    입시결과 파일을 간단히 파싱하여 DB에 저장하는 함수.
    """
    pdf_document = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    title = os.path.basename(file_path)
    save_parsed_pdf_to_db(pdf_document, doc_type, doc_category, title)

def extract_toc(pdf_document, page_gap):
    toc_mapping = {}
    contentPage = None

    for idx, page in enumerate(pdf_document):
        text = page['text']
        if "CONTENTS" in text:
            contentPage = idx + 1
            pdf_document[idx]['metadata']['title'] = "CONTENTS"
            break

    if contentPage is None:
        raise ValueError("목차 페이지를 찾을 수 없습니다. PDF에 'CONTENTS'가 포함되어 있는지 확인하세요.")

    toc_text = pdf_document[contentPage - 1]['text']
    toc_lines = toc_text.split("\n")

    for line in toc_lines:
        clean_line = re.sub(r"[^\w\s]", "", line).strip()
        match = re.search(r"(.*?)\s+(\d+)$", clean_line)
        if match:
            title = match.group(1).strip()
            start_page = int(match.group(2).strip()) + page_gap
            toc_mapping[start_page] = title

    return toc_mapping

def add_titles_to_pages(pdf_document, toc_mapping):
    for idx, page in enumerate(pdf_document):
        current_page = idx + 1
        metadata = page['metadata']

        for start_page in sorted(toc_mapping.keys()):
            if current_page >= start_page:
                next_page_start = min([p for p in toc_mapping.keys() if p > start_page], default=None)
                if next_page_start is None or current_page < next_page_start:
                    metadata['title'] = toc_mapping[start_page]
                    break

def modify_text_based_on_metadata(pdf_document):
    for idx, page in enumerate(pdf_document):
        current_page = idx + 1
        metadata = page['metadata']
        text = page['text']
        
        title = metadata.get('title', '')
        if title:
            text = f"**{title} 문서입니다. - 중요! 이 문서 전체는 {title}에 해당하는 정보입니다.**\n\n" + text
            text = f"**{title}**\n" + text

            pdf_document[idx]['text'] = text

def save_parsed_pdf_to_db(pdf_document, doc_type, doc_category, title):
    model_class = get_model_class(doc_type)

    for idx, page in enumerate(pdf_document):
        current_page = idx + 1

        page_text = page['text']

        model_class.objects.create(
            title=title,
            content=page_text,
            page=current_page,
            category=doc_category
        )