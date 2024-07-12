from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document1, Document2, Document3
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from drf_yasg.utils       import swagger_auto_schema
from drf_yasg             import openapi
import tempfile
import os
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def main(request):
    return render(request, "maruegg/upload_html_file.html")

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes

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
            description='Category of the document (모집요강, 입시결과, 기출문제)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제']
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
        
        if doc_category not in ["모집요강", "입시결과", "기출문제"]:
            return JsonResponse({"error": "Invalid category provided"}, status=400)

        model_class = None
        if doc_type == "수시":
            model_class = Document1
        elif doc_type == "정시":
            model_class = Document2
        elif doc_type == "편입학":
            model_class = Document3

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_file.read())
            temp_file_path = temp_file.name
        try:
            logger.debug(f"Deleting all documents from {model_class.__name__} in the database")
            model_class.objects.all().delete()

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

            title = html_file.name
            parts = split_text(tables_text, max_length=1500)
            for i, part in enumerate(parts):
                model_class.objects.create(
                    title=title,
                    content=part,
                    part=i+1,
                    category=doc_category
                )
                logger.debug(f"Document part {i+1} created with content: {part[:100]}...")
        finally:
            os.remove(temp_file_path)
        return render(request, "maruegg/upload_html_file.html", {"message": "Document uploaded and parsed successfully"})
    return HttpResponse("Invalid request", status=400)

def split_text(text, max_length=1500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

from rest_framework.parsers import FormParser

@swagger_auto_schema(
    method='post',
    operation_description="questionType(수시, 정시, 편입학), questionCategory(모집요강, 입시결과, 기출문제), question(질문)을 기반으로 rag방식 llm모델의 답변을 요청하는 api",
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
            description='Category of the question (모집요강, 입시결과, 기출문제)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제']
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
        question_category = request.POST.get("questionCategory")

        logger.debug(f"Received question: {question}")
        logger.debug(f"Received question type: {question_type}")
        logger.debug(f"Received question category: {question_category}")

        if not question:
            return JsonResponse({"error": "No content provided"}, status=400)

        if question_type not in ["수시", "정시", "편입학"]:
            return JsonResponse({"error": "Invalid question type provided"}, status=400)
        
        if question_category not in ["모집요강", "입시결과", "기출문제"]:
            return JsonResponse({"error": "Invalid question category provided"}, status=400)

        start_time = time.time()
        context = get_relevant_documents(question_type, question_category, question)
        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Document retrieval took {retrieval_time:.2f} seconds")

        if not context:
            logger.debug("No relevant documents found.")
            return JsonResponse({"error": "No relevant documents found"}, status=404)
        
        logger.debug(f"Context for question: {context[:100]}...")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQ: {question}\nA:"}
            ]
        )
        end_time = time.time()
        completion_time = end_time - start_time
        logger.debug(f"OpenAI completion took {completion_time:.2f} seconds")
        
        answer = response.choices[0].message.content.strip()
        
        return JsonResponse({"questionType": question_type, "questionCategory": question_category, "answer": answer})

    return render(request, "maruegg/ask_question.html")


def get_relevant_documents(question_type, question_category, question, max_docs=5):
    model_class = None
    if question_type == "수시":
        model_class = Document1
    elif question_type == "정시":
        model_class = Document2
    elif question_type == "편입학":
        model_class = Document3

    documents = model_class.objects.filter(category=question_category)
    if not documents.exists():
        return ""

    doc_contents = [doc.content for doc in documents]
    vectorizer = TfidfVectorizer().fit_transform([question] + doc_contents)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity(vectors)
    scores = cosine_matrix[0][1:]
    
    top_indices = scores.argsort()[-max_docs:][::-1]
    relevant_docs = [doc_contents[i] for i in top_indices]
    context = "\n\n".join(relevant_docs)
    
    return context

def ask_question_model(request):
    pass