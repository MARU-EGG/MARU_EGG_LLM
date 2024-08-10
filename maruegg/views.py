from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document1, Document2, Document3
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes
from rest_framework.parsers import FormParser
from drf_yasg import openapi
from bs4 import BeautifulSoup
from openai import OpenAI
import tempfile
import os
import time
import logging
import fitz
import camelot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

MEDIA_DOCUMENT_URL = os.path.join(settings.MEDIA_ROOT, 'documents')
MEDIA_FILES_URL = os.path.join(settings.MEDIA_ROOT, 'files')


def main(request):
    return render(request, "maruegg/upload_html_file.html")


@swagger_auto_schema(
    method='post',
    operation_description="LLM모델이 답변할 기반이 되는 html파일 업로드 api입니다. type, category에 따라 db에 저장되는 위치가 달라집니다.",
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
            'html_file',
            openapi.IN_FORM,
            description='HTML file to be uploaded',
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
def upload_html(request):
    if request.method == "POST" and 'html_file' in request.FILES:
        html_file = request.FILES["html_file"]
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

        filename = f"{doc_type}_{doc_category}.html"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_file.read())
            temp_file_path = temp_file.name
        
        try:
            save_file(html_file, filename)

            logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
            model_class.objects.filter(category=doc_category).delete()

            with open(temp_file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')

            title = filename
            tables = soup.find_all('table')
            for page_num, table in enumerate(tables, start=1):
                rows = table.find_all('tr')
                page_text = ""
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = "\t".join(cell.get_text(strip=True) for cell in cells)
                    page_text += row_text + "\n"
                page_text += "\n"

                model_class.objects.create(
                    title=title,
                    content=page_text,
                    page=page_num,
                    category=doc_category
                )
                logger.debug(f"Document from page {page_num} created with content: {page_text[:100]}...")
        finally:
            os.remove(temp_file_path)
        return render(request, "maruegg/upload_html_file.html", {"message": "Document uploaded and parsed successfully"})
    return HttpResponse("Invalid request", status=400)

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

        filename = f"{doc_type}_{doc_category}.pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in pdf_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        try:
            save_file(pdf_file, filename)

            logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
            model_class.objects.filter(category=doc_category).delete()
            pdf_document = fitz.open(temp_file_path)

            title = filename
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                page_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                page_text = ' '.join(page_lines)

                tables = camelot.read_pdf(temp_file_path, pages=str(page_num+1), flavor='stream')
                if tables:
                    for table in tables:
                        table_text = table.df.to_csv(index=False)
                        page_text += '\n' + table_text

                model_class.objects.create(
                    title=title,
                    content=page_text,
                    page=page_num + 1,
                    category=doc_category
                )
                logger.debug(f"Document from page {page_num + 1} created with content: {page_text[:100]}...")
        finally:
            os.remove(temp_file_path)
        return render(request, "maruegg/upload_html_file.html", {"message": "Document uploaded and parsed successfully"})
    return HttpResponse("Invalid request", status=400)


@swagger_auto_schema(
    method='post',
    operation_description="questionType(수시, 정시, 편입학), questionCategory(모집요강, 입시결과, 기출문제, 대학생활, 면접/실기), question(질문)을 기반으로 rag방식 llm모델의 답변을 요청하는 api",
    manual_parameters=[
        openapi.Parameter(
            'questionType',
            openapi.IN_FORM,
            description='Type of the question (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'questionCategory',
            openapi.IN_FORM,
            description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['', '모집요강', '입시결과', '기출문제', '대학생활', '면접/실기'],
            default=''
        ),
        openapi.Parameter(
            'question',
            openapi.IN_FORM,
            description='The actual question content',
            required=True,
            type=openapi.TYPE_STRING
        ),
    ],
    responses={200: 'Success', 400: 'Invalid request', 404: 'No relevant documents found'}
)
@csrf_exempt
@api_view(['POST'])
@parser_classes([FormParser])
def ask_question_api(request):
    if request.method == "POST":
        question = request.POST.get("question")
        question_type = request.POST.get("questionType")
        question_category = request.POST.get("questionCategory", "")

        logger.debug(f"Received question: {question}")
        logger.debug(f"Received question type: {question_type}")
        logger.debug(f"Received question category: {question_category}")

        if not question:
            return JsonResponse({"error": "No content provided"}, status=400)

        if question_type not in ["수시", "정시", "편입학"]:
            return JsonResponse({"error": "Invalid question type provided"}, status=400)
        
        start_time = time.time()
        context, references = get_relevant_documents(question_type, question_category, question)
        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Document retrieval took {retrieval_time:.2f} seconds")

        if not context:
            logger.debug("No relevant documents found.")
            return JsonResponse({"error": "No relevant documents found"}, status=404)
        
        logger.debug(f"Context for question: {context[:100]}...")
        
        start_time = time.time()
        
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": f"Context: {context}\n\nQ: {question}\nA:"}
        #     ]
        # )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user", 
                    "content": (
                        f"Context: {context}\n\n"
                        f"Q: {question}\n"
                        "A: Please provide a helpful and relevant answer based only on the provided context. "
                        "If the context does not contain sufficient information to answer the question accurately, "
                        "respond with '해당 내용에 대한 정보는 존재하지 않습니다. 정확한 내용은 입학지원팀에 문의해주세요.'."
                    )
                }
            ],
            max_tokens=150
        )
        end_time = time.time()
        completion_time = end_time - start_time
        logger.debug(f"OpenAI completion took {completion_time:.2f} seconds")
        
        answer = response.choices[0].message.content.strip()
        
        if question_category == "":
            question_category = references[0]['category']

        base_url = "http://127.0.0.1:8000" if settings.DEBUG else "http://3.37.12.249"

        references_response = [
            {
                "title": ref['title'],
                "link": f"{base_url}/media/documents/{ref['title']}#page={ref['page']}"
            } for ref in references
        ]

        return JsonResponse({
            "questionType": question_type, 
            "questionCategory": question_category, 
            "answer": answer,
            "references": references_response
        })

    return render(request, "maruegg/ask_question.html")

def get_relevant_documents(question_type, question_category, question, max_docs=5):
    model_class = None
    if question_type == "수시":
        model_class = Document1
    elif question_type == "정시":
        model_class = Document2
    elif question_type == "편입학":
        model_class = Document3

    if question_category:
        documents = model_class.objects.filter(category=question_category)
    else:
        documents = model_class.objects.all()

    if not documents.exists():
        return "", []

    doc_contents = [(doc.title, doc.content, doc.page, doc.category) for doc in documents]
    contents = [content for _, content, _, _ in doc_contents]
    vectorizer = TfidfVectorizer().fit_transform([question] + contents)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity(vectors)
    scores = cosine_matrix[0][1:]
    
    top_indices = scores.argsort()[-max_docs:][::-1]
    relevant_docs = [doc_contents[i] for i in top_indices]
    
    context = "\n\n".join([content for _, content, _, _ in relevant_docs])
    references = [{"title": title, "page": page, "category": category} for title, _, page, category in relevant_docs]
    
    return context, references


# Delete APIs
@swagger_auto_schema(
    method='delete',
    operation_description="DB내 데이터를 삭제합니다. (특정 type, category별 또는 전체)",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to delete (수시, 정시, 편입학)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the documents to delete (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_documents(request):
    doc_type = request.GET.get("type")
    doc_category = request.GET.get("category")
    
    model_classes = {
        "수시": Document1,
        "정시": Document2,
        "편입학": Document3
    }
    
    if doc_type and doc_type not in model_classes:
        return JsonResponse({"error": "Invalid type provided"}, status=400)
    
    if doc_category and doc_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
        return JsonResponse({"error": "Invalid category provided"}, status=400)
    
    if doc_type:
        model_class = model_classes[doc_type]
        queryset = model_class.objects.all()
        if doc_category:
            queryset = queryset.filter(category=doc_category)
        queryset.delete()
    else:
        for model_class in model_classes.values():
            queryset = model_class.objects.all()
            if doc_category:
                queryset = queryset.filter(category=doc_category)
            queryset.delete()
    
    return JsonResponse({"message": "Documents deleted successfully"}, status=200)

# Retrieve APIs
@swagger_auto_schema(
    method='get',
    operation_description="DB내 데이터를 반환합니다. (특정 type, category별 또는 전체)",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to retrieve (수시, 정시, 편입학)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the documents to retrieve (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['GET'])
def retrieve_documents(request):
    doc_type = request.GET.get("type")
    doc_category = request.GET.get("category")
    
    model_classes = {
        "수시": Document1,
        "정시": Document2,
        "편입학": Document3
    }
    
    if doc_type and doc_type not in model_classes:
        return JsonResponse({"error": "Invalid type provided"}, status=400)
    
    if doc_category and doc_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
        return JsonResponse({"error": "Invalid category provided"}, status=400)
    
    documents = {}
    if doc_type:
        model_class = model_classes[doc_type]
        queryset = model_class.objects.all()
        if doc_category:
            queryset = queryset.filter(category=doc_category)
        documents[doc_type] = list(queryset.values())
    else:
        for doc_type, model_class in model_classes.items():
            queryset = model_class.objects.all()
            if doc_category:
                queryset = queryset.filter(category=doc_category)
            documents[doc_type] = list(queryset.values())
    
    return JsonResponse({"documents": documents}, status=200)

# main source
def upload_html_show(request):
    if request.method == "GET":
        return render(request, "maruegg/upload_html_file.html")
    if request.method == "POST" and request.FILES.get("html_file"):
        html_file = request.FILES["html_file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_file.read())
            temp_file_path = temp_file.name
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')

            tables_text = ""
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = "\t".join(cell.get_text(strip=True) for cell in cells)
                    tables_text += row_text + "\n"
                tables_text += "\n"
        finally:
            os.remove(temp_file_path)
        return render(request, "maruegg/upload_html_file.html", {"tables_text": tables_text})
    return HttpResponse("Invalid request", status=400)

def save_file(file, filename):
    file_path = os.path.join(MEDIA_DOCUMENT_URL, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path