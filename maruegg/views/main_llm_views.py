import os
import tempfile
import logging
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from openai import OpenAI
import fitz
import camelot
from ..models import Document1, Document2, Document3

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

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

        if not question:
            return JsonResponse({"error": "No content provided"}, status=400)
        if question_type not in ["수시", "정시", "편입학"]:
            return JsonResponse({"error": "Invalid question type provided"}, status=400)

        context, references = get_relevant_documents(question_type, question_category, question)
        if not context:
            logger.debug("No relevant documents found.")
            return JsonResponse({"error": "No relevant documents found"}, status=404)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user", 
                    "content": (
                        f"Context: {context}\n\n"
                        f"Q: {question}\n"
                        "A: Based on the provided context, please answer the question as accurately as possible. "
                        "If the context does not provide enough information, give a brief, relevant response using only the available details. "
                        "If you cannot find any relevant information in the context, reply with '해당 내용에 대한 정보는 존재하지 않습니다. "
                        "정확한 내용은 입학지원팀에 문의해주세요.'."
                    )
                }
            ],
            max_tokens=400,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        
        if question_category == "":
            question_category = references[0]['category']

        base_url = "http://127.0.0.1:8000" if settings.DEBUG else "https://marueggllmserver.com"
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
    return JsonResponse({"error": "Invalid request method"}, status=405)

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

