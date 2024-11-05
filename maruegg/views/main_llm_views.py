import os
import time
import logging
import uuid
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from openai import OpenAI
from ..models import Document1, Document2, Document3, Prompt
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

MEDIA_DOCUMENT_URL = os.path.join(settings.MEDIA_ROOT, 'documents')
MEDIA_FILES_URL = os.path.join(settings.MEDIA_ROOT, 'files')

logger = logging.getLogger(__name__)

import hashlib

def generate_metadata_hash(metadata):
    """Generate a unique hash for a document based on its metadata."""
    metadata_string = "".join(str(value) for key, value in sorted(metadata.items()) if key != "doc_id")
    return hashlib.sha256(metadata_string.encode('utf-8')).hexdigest()

def create_multi_vector_retriever(vectorstore, doc_contents):
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    existing_metadata = retriever.vectorstore.get(include=["metadatas"])
    existing_hashes = set(
        generate_metadata_hash(metadata) for metadata in existing_metadata["metadatas"]
    )

    existing_docs = list(zip(
        [metadata[id_key] for metadata in existing_metadata["metadatas"]],
        [metadata for metadata in existing_metadata["metadatas"]]
    ))

    new_docs = []
    new_doc_ids = []

    for doc_content in doc_contents:
        metadata = {
            "title": doc_content[0],
            "page": doc_content[2],
            "category": doc_content[3],
            "content": doc_content[1],
        }
        doc_hash = generate_metadata_hash(metadata)

        if doc_hash not in existing_hashes:
            doc_id = str(uuid.uuid4())
            new_doc_ids.append(doc_id)
            metadata[id_key] = doc_id
            new_docs.append(
                Document(
                    page_content=doc_content[1],
                    metadata=metadata
                )
            )
            existing_docs.append((doc_id, metadata))
        else:
            logger.debug("Document with similar content already exists. Skipping.")

    if new_docs:
        retriever.vectorstore.add_documents(new_docs)

    retriever.docstore.mset(existing_docs)

    return retriever

def split_text_types(docs):
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if isinstance(doc, list):
            doc = " ".join(str(item) for item in doc)
        texts.append(doc)
    return {"texts": texts}

def prompt_func(data_dict, question_type, question_category):
    try:
        prompt_instance = Prompt.objects.get(
            question_type=question_type,
            question_category=question_category
        )
        gpt_prompt = prompt_instance.prompt_text
        print(gpt_prompt)
    except Prompt.DoesNotExist:
        logger.error("No prompt found for the given question type and category.")
        return JsonResponse({"error": "No prompt found for the given question type and category"}, status=404)

    try:
        formatted_texts = "\n".join(str(text) for text in data_dict["context"]["texts"])
    except Exception as e:
        logger.error(f"Error in formatting texts: {e}")
        raise

    messages = []

    text_message = {
        "type": "text",
        "text": (
            gpt_prompt +
            f"## User-provided question:\n**{data_dict['question']}**\n\n"
            "### Text and / or tables:\n"
            f"```\n{formatted_texts}\n```\n\n"
            # "Please provide your answer in the following format:\n"
            # "- **Main Point 1**: Detail...\n"
            # "- **Main Point 2**: Detail...\n"
            # "- **Main Point 2**: Detail...\n"
            # "- **Main Point 2**: Detail...\n"
            # "- **Main Point 2**: Detail...\n"
        ),
    }

    messages.append(text_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain(retriever, question_type, question_category):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=500)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=2048)

    chain = (
        {
            "context": multi_query_retriever | RunnableLambda(split_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(lambda data_dict: prompt_func(data_dict, question_type, question_category))
        | model
        | StrOutputParser()
    )
    return chain

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
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기'],
            default='모집요강'
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
    question = request.POST.get("question")
    question_type = request.POST.get("questionType")
    question_category = request.POST.get("questionCategory")

    if not question:
        return JsonResponse({"error": "No content provided"}, status=400)

    if question_type not in ["수시", "정시", "편입학"]:
        return JsonResponse({"error": "Invalid question type provided"}, status=400)

    start_time = time.time()
    retriever, references = get_relevant_documents(question_type, question_category, question)
    retrieval_time = time.time() - start_time

    if not retriever:
        logger.debug("No relevant documents found.")
        return JsonResponse({"error": "No relevant documents found"}, status=404)

    try:
        chain_multimodal_rag = multi_modal_rag_chain(retriever, question_type, question_category)
        response = chain_multimodal_rag.invoke({"question": question})
        completion_time = time.time() - start_time

    except Exception as e:
        logger.error(f"Error during RAG processing: {e}")
        return JsonResponse({"error": "An error occurred during processing"}, status=500)

    if not response:
        logger.debug("No relevant documents found or no response generated.")
        return JsonResponse({"error": "No relevant documents found"}, status=404)

    answer = response.strip()
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

def get_relevant_documents(question_type, question_category, question, max_docs=5):
    model_class = None
    if question_type == "수시":
        model_class = Document1
    elif question_type == "정시":
        model_class = Document2
    elif question_type == "편입학":
        model_class = Document3

    class_name = model_class.__name__

    category_mapping = {
        "모집요강": "admission_guideline",
        "입시결과": "admission_results",
        "기출문제": "past_exams",
        "대학생활": "campus_life",
        "면접/실기": "interview_practice"
    }

    if not question_category:
        categories = ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]
    else:
        categories = [question_category]

    retrievers = {}

    for category in categories:
        english_category = category_mapping.get(category, category)
        documents = model_class.objects.filter(category=category)
        if documents.exists():
            doc_contents = [(doc.title, doc.content, doc.page, doc.category) for doc in documents]
            embedding_function = OpenAIEmbeddings()

            persist_directory = f"vectorDB/{class_name}/{english_category}_vectorDB"

            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = Chroma(
                collection_name=f"{english_category}_collection",
                embedding_function=embedding_function,
                persist_directory=persist_directory
            )
            retrievers[category] = create_multi_vector_retriever(vectorstore, doc_contents)

    if question_category:
        retriever = retrievers.get(question_category)
        if retriever:
            relevant_docs = retriever.vectorstore.similarity_search_with_score(question, k=max_docs)
        else:
            return "", []
    else:
        relevant_docs = []
        for category, retriever in retrievers.items():
            category_docs = retriever.vectorstore.similarity_search_with_score(question, k=3)
            relevant_docs.extend(category_docs)

    if not relevant_docs:
        return "", []

    references = []
    for doc in relevant_docs[:3]:
        metadata = doc[0].metadata
        references.append({
        "title": metadata.get("title", "Unknown Title"),
        "page": metadata.get("page", "Unknown Page"),
        "category": metadata.get("category", "Unknown Category")
        })
    return retriever, references