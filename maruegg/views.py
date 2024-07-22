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
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import tempfile
import os
import numpy as np
import time
import logging
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes
from rest_framework.parsers import FormParser
import fitz
import camelot

# LangChain imports
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# import requests
# from transformers import AutoTokenizer, AutoModel
# import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

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

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_file.read())
            temp_file_path = temp_file.name

        try:
            logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
            model_class.objects.filter(category=doc_category).delete()

            with open(temp_file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')

            title = html_file.name
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

def split_text(text, max_length=1000):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if len('\n'.join(current_chunk + [paragraph])) <= max_length:
            current_chunk.append(paragraph)
        else:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

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

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in pdf_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        try:
            logger.debug(f"Deleting documents from {model_class.__name__} in the database with category {doc_category}")
            model_class.objects.filter(category=doc_category).delete()

            # PDF 파일 열기
            pdf_document = fitz.open(temp_file_path)

            title = pdf_file.name
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")

                # 공백 라인을 제거하고 라인들을 연결하여 텍스트를 정리
                page_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                page_text = ' '.join(page_lines)

                # 페이지에서 표 추출
                tables = camelot.read_pdf(temp_file_path, pages=str(page_num+1), flavor='stream')
                if tables:
                    for table in tables:
                        table_text = table.df.to_csv(index=False)
                        page_text += '\n' + table_text

                # 페이지 텍스트를 하나의 레코드로 저장
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
            model="gpt-4o-mini",
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

    if question_category:
        documents = model_class.objects.filter(category=question_category)
    else:
        documents = model_class.objects.all()

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

# @swagger_auto_schema(
#     method='post',
#     operation_description="LLM을 사용하여 질문에 답변을 생성하는 API입니다. 질문을 보내면 데이터베이스에서 관련 문서를 검색하고 답변을 생성합니다.",
#     manual_parameters=[
#         openapi.Parameter(
#             'questionType',
#             openapi.IN_FORM,
#             description='Type of the question (수시, 정시, 편입학)',
#             required=True,
#             type=openapi.TYPE_STRING,
#             enum=['수시', '정시', '편입학']
#         ),
#         openapi.Parameter(
#             'questionCategory',
#             openapi.IN_FORM,
#             description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
#             required=False,
#             type=openapi.TYPE_STRING,
#             enum=['', '모집요강', '입시결과', '기출문제', '대학생활', '면접/실기'],
#             default=''
#         ),
#         openapi.Parameter(
#             'question',
#             openapi.IN_FORM,
#             description='The actual question content',
#             required=True,
#             type=openapi.TYPE_STRING
#         ),
#     ],
#     responses={200: 'Success', 400: 'Invalid request', 404: 'No relevant documents found'}
# )
# @csrf_exempt
# @api_view(['POST'])
# @parser_classes([FormParser])
# def ask_question_model(request):
#     if request.method == "POST":
#         question = request.POST.get("question")
#         question_type = request.POST.get("questionType")
#         question_category = request.POST.get("questionCategory", "")

#         logger.debug(f"Received question: {question}")
#         logger.debug(f"Received question type: {question_type}")
#         logger.debug(f"Received question category: {question_category}")

#         if not question:
#             return JsonResponse({"error": "No content provided"}, status=400)

#         if question_type not in ["수시", "정시", "편입학"]:
#             return JsonResponse({"error": "Invalid question type provided"}, status=400)

#         # Get relevant documents from the database
#         context = get_relevant_documents(question_type, question_category, question)
#         if not context:
#             logger.debug("No relevant documents found.")
#             return JsonResponse({"error": "No relevant documents found"}, status=404)

#         # Convert context to a list of Document objects
#         docs = [Document(page_content=content) for content in context.split("\n\n")]

#         # Set up LangChain components
#         embeddings = HuggingFaceEmbeddings(
#             model_name='BAAI/bge-m3', 
#             model_kwargs={'device':'mps'}, 
#             encode_kwargs={'normalize_embeddings':True}
#         )
#         vectorstore = Chroma.from_documents(docs, embeddings, persist_directory='vectorstore')
#         vectorstore.persist()
        
#         # Set up the retriever
#         retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

#         # Set up the prompt template
#         template = '''당신은 친절한 챗봇으로서 명지대학교 입시 관련 질문자 요청에 최대한 자세하고 친절하게 답변해야 합니다. 모든 대답은 한국어(Korean)으로 대답해 주세요. 만약 질문이 명지대학교 입시와 관련이 없다면 "해당 내용은 답변할 수 없습니다!"라고 대답해 주세요:
#         {context}

#         Question: {question}
#         '''
#         prompt = ChatPromptTemplate.from_template(template)

#         # Define the chain components
#         def format_docs(docs):
#             return '\n\n'.join([d.page_content for d in docs])

#         rag_chain = (
#             {'context': retriever | format_docs, 'question': RunnablePassthrough()}
#             | prompt
#             | ChatOllama(model="EEVE-Korean-10.8B:latest", base_url="https://e9b5-128-134-33-155.ngrok-free.app")
#             | StrOutputParser()
#         )

#         # Run the chain
#         answer = rag_chain.invoke(question)

#         logger.debug(f"Generated answer: {answer}")

#         return JsonResponse({"questionType": question_type, "questionCategory": question_category, "answer": answer})

#     return HttpResponse("Invalid request", status=400)


# Delete APIs
@swagger_auto_schema(
    method='delete',
    operation_description="DB내 모든 데이터를 삭제합니다.",
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_documents_all(request):
    if request.method == 'DELETE':
        Document1.objects.all().delete()
        Document2.objects.all().delete()
        Document3.objects.all().delete()
        return JsonResponse({"message": "All documents deleted successfully"}, status=200)
    return JsonResponse({"error": "Invalid request"}, status=400)


@swagger_auto_schema(
    method='delete',
    operation_description="DB내 특정 type의 데이터를 삭제합니다.",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to delete (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_documents_by_type(request):
    doc_type = request.GET.get("type")

    if doc_type not in ["수시", "정시", "편입학"]:
        return JsonResponse({"error": "Invalid type provided"}, status=400)

    model_class = None
    if doc_type == "수시":
        model_class = Document1
    elif doc_type == "정시":
        model_class = Document2
    elif doc_type == "편입학":
        model_class = Document3

    model_class.objects.all().delete()
    return JsonResponse({"message": f"All {doc_type} documents deleted successfully"}, status=200)


@swagger_auto_schema(
    method='delete',
    operation_description="DB내 특정 type & category의 데이터를 삭제합니다.",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to delete (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the documents to delete (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_documents_by_type_and_category(request):
    doc_type = request.GET.get("type")
    doc_category = request.GET.get("category")

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

    model_class.objects.filter(category=doc_category).delete()
    return JsonResponse({"message": f"All {doc_type} documents in category {doc_category} deleted successfully"}, status=200)


# Retrieve APIs
@swagger_auto_schema(
    method='get',
    operation_description="DB내 모든 데이터를 반환합니다.",
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['GET'])
def retrieve_documents_all(request):
    documents1 = list(Document1.objects.all().values())
    documents2 = list(Document2.objects.all().values())
    documents3 = list(Document3.objects.all().values())

    return JsonResponse({"documents1": documents1, "documents2": documents2, "documents3": documents3}, status=200)


@swagger_auto_schema(
    method='get',
    operation_description="DB내 특정 type의 데이터를 반환합니다.",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to retrieve (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['GET'])
def retrieve_documents_by_type(request):
    doc_type = request.GET.get("type")

    if doc_type not in ["수시", "정시", "편입학"]:
        return JsonResponse({"error": "Invalid type provided"}, status=400)

    model_class = None
    if doc_type == "수시":
        documents = list(Document1.objects.all().values())
    elif doc_type == "정시":
        documents = list(Document2.objects.all().values())
    elif doc_type == "편입학":
        documents = list(Document3.objects.all().values())

    return JsonResponse({"documents": documents}, status=200)


@swagger_auto_schema(
    method='get',
    operation_description="DB내 특정 type & category의 데이터를 반환합니다.",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to retrieve (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the documents to retrieve (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['GET'])
def retrieve_documents_by_type_and_category(request):
    doc_type = request.GET.get("type")
    doc_category = request.GET.get("category")

    if doc_type not in ["수시", "정시", "편입학"]:
        return JsonResponse({"error": "Invalid type provided"}, status=400)

    if doc_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
        return JsonResponse({"error": "Invalid category provided"}, status=400)

    model_class = None
    if doc_type == "수시":
        documents = list(Document1.objects.filter(category=doc_category).values())
    elif doc_type == "정시":
        documents = list(Document2.objects.filter(category=doc_category).values())
    elif doc_type == "편입학":
        documents = list(Document3.objects.filter(category=doc_category).values())

    return JsonResponse({"documents": documents}, status=200)



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