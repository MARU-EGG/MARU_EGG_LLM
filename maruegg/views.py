from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup
import tempfile
import os
from .models import Document
import logging
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


@csrf_exempt
def test(request):
    if request.method == "POST":
        received_value = request.POST.get("string_value", "")
        
        logger.debug(f"Received value: {received_value}")
        
        return JsonResponse({"message": "요청이 잘 들어왔습니다", "received_value": received_value})
    return HttpResponse("Invalid request", status=400)


@csrf_exempt
def api_ask_question(request):
    if request.method == "POST":
        question_type = request.POST.get("questionType")
        question_category = request.POST.get("questionCategory")
        content = request.POST.get("content")

        logger.debug(f"Received questionType: {question_type}")
        logger.debug(f"Received questionCategory: {question_category}")

        if not content:
            return JsonResponse({"error": "No content provided"}, status=400)

        logger.debug(f"Searching for content: {content}")
        
        start_time = time.time()
        context = get_relevant_documents(content)
        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Document retrieval took {retrieval_time:.2f} seconds")

        if not context:
            logger.debug("No relevant documents found.")
            return JsonResponse({"error": "No relevant documents found"}, status=404)
        
        logger.debug(f"Context for content: {context[:100]}...")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQ: {content}\nA:"}
            ]
        )
        end_time = time.time()
        completion_time = end_time - start_time
        logger.debug(f"OpenAI completion took {completion_time:.2f} seconds")
        
        answer = response.choices[0].message.content.strip()
        
        return JsonResponse({"questionType": question_type, "questionCategory": question_category, "answer": answer})

    return JsonResponse({"error": "Invalid request"}, status=400)




def split_text(text, max_length=500):
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

def main(request):
    return render(request, "maruegg/upload_html_file.html")

@csrf_exempt
def upload_html(request):
    if request.method == "POST" and request.FILES.get("html_file"):
        html_file = request.FILES["html_file"]

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_file.read())
            temp_file_path = temp_file.name

        try:
            logger.debug("Deleting all documents from the database")
            Document.objects.all().delete()

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
            parts = split_text(tables_text, max_length=500)
            for i, part in enumerate(parts):
                Document.objects.create(title=title, content=part, part=i+1)
                logger.debug(f"Document part {i+1} created with content: {part[:100]}...")  # 처음 100자만 출력

        finally:
            os.remove(temp_file_path)

        return render(request, "maruegg/upload_html_file.html", {"message": "Document uploaded and parsed successfully"})

    return HttpResponse("Invalid request", status=400)

def get_relevant_documents(question, max_docs=5):
    documents = Document.objects.all()
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

@csrf_exempt
def ask_question_api(request):
    question = None
    answer = None

    if request.method == "POST":
        question = request.POST.get("question")
        
        logger.debug(f"Searching for question: {question}")
        
        start_time = time.time()
        context = get_relevant_documents(question)
        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Document retrieval took {retrieval_time:.2f} seconds")

        if not context:
            logger.debug("No relevant documents found.")
            return HttpResponse("No relevant documents found.")
        
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
        
    return render(request, "maruegg/ask_question.html", {"question": question, "answer": answer})

def ask_question_model(request):
    pass