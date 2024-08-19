import os
import time
import logging
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from openai import OpenAI
from ..models import Document1, Document2, Document3
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

    # 기존 문서의 메타데이터와 해시값 가져오기
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
        # 메타데이터 해시 생성
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
            # 새로운 문서도 추가
            existing_docs.append((doc_id, metadata))
        else:
            logger.debug("Document with similar content already exists. Skipping.")

    # 새로운 문서가 있을 경우 추가하고, 모든 문서를 다시 mset에 저장
    if new_docs:
        retriever.vectorstore.add_documents(new_docs)
    
    # 기존 문서와 새로운 문서를 mset에 저장
    retriever.docstore.mset(existing_docs)

    return retriever

# 문서 텍스트를 추출하는 함수
def split_text_types(docs):
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if isinstance(doc, list):
            doc = " ".join(str(item) for item in doc)
        texts.append(doc)
    return {"texts": texts}

# 프롬프트 생성 함수
def prompt_func(data_dict):
    try:
        formatted_texts = "\n".join(str(text) for text in data_dict["context"]["texts"])  # Ensure all items are strings
    except Exception as e:
        logger.error(f"Error in formatting texts: {e}")
        raise

    messages = []
    
    # GPT 프롬프트 설정
    gpt_prompt = (
        "명지대학교 입학 관련된 상담을 진행하기 위한 챗봇입니다.\n\n"
        "가이드라인:\n"
        "명지대학교에는 크게 학생부교과, 학생부종합, 실기/실적, 기타 총 네 가지의 입학전형을 가집니다.\n"
        "학생부교과안에는 면접이 포함된 전형과 포함되지 않는 전형으로 나뉘며 면접을 보지 않는 전형은 학교장추천전형, 기회균형전형, 특성화고교전형이 있고, "
        "면접이 포함되는 전형은 교과면접전형, 만학도전형, 특성화고등졸재직자전형, 특수교육대상자전형이 있습니다.\n"
        "학생부종합 안에는 면접이 포함된 전형과 포함되지 않는 전형으로 나뉘고, 면접을 포함하는 전형은 명지인재면접전형, 크리스천리더전형이며 "
        "면접이 포함되지 않는 전형은 명지인재서류전형, 사회적배려대상자전형, 농어촌학생전형입니다.\n"
        "실기/실적은 실기를 보는 실기우수자전형, 면접과 실적을 확인하는 특기자전형이 있습니다.\n"
        "기타는 재외국민전형, 전교육과정국외이수자전형, 북한이탈주민전형이 있으며 면접 100% 심사를 통해 선발됩니다.\n\n"
        "지시사항:\n"
        "전형을 줄여서 말해도 구분할 수 있어야합니다. (예: 학교장추천전형: 학생부교과(학교장추천전형), 명지인재면접전형: 학생부종합(명지인재면접전형))\n"
        "구분하기 애매한 전형의 경우에는 전형을 확실히 말해달라는 멘트와 함께 전형을 보여줄 수 있어야합니다. "
        "(예: 학생부교과전형 설명해줘 - 학생부 교과 전형에는 학교장추천전형, 교과면접전형, 기회균형전형, 특성화고교전형, 만학도전형, 특성화고등졸재직자전형이 있습니다. "
        "여기서 어떤 전형에 대해 설명해드릴까요?)\n"
        "지원자격에 대한 모든 기준을 말해야합니다. "
        "(예: 사회적배려대상자전형의 지원자격의 경우 1)직업군인, 경찰, 소방, 교정공무원의 자녀, 2) 다자녀 가정의 자녀, 3)북한이탈주민이나 제3국 출생 북한이탈주민 가정의 자녀, "
        "4)다문화 가정의 자녀, 5)의사상자 및 그의 자녀가 지원할 수 있습니다.)\n"
        "특정 전형에 대한 서류를 요청하는 경우 가능한 모든 서류를 말해야합니다. "
        "(예: 사회적배려대상자전형의 경우 국내 고등학교 졸업(예정)자는 학교생활기록부와 입학원서, 검정고시출신자는 입학원서, "
        "고등학교 졸업학력 검정고시 성적증명서 및 합격증명서, 학교생활기록부 대체서식, 국외고등학교졸업(예정)자는 입학원서와 국외고 성적증명서 및 졸업(예정)증명서, "
        "학교생활기록부 대체 서식이 필요합니다.)\n"
        "모든 대답의 마지막에는 ‘더욱 자세한 상담을 원하시면 명지대학교 입학처 02-300-1799,1800으로 전화주시길 바랍니다.’라는 멘트를 붙여야합니다.\n\n"
    )
    
    # 사용자의 질문과 컨텍스트를 포함하는 메시지 생성
    text_message = {
        "type": "text",
        "text": (
            gpt_prompt +
            f"## User-provided question:\n**{data_dict['question']}**\n\n"
            "### Text and / or tables:\n"
            f"```\n{formatted_texts}\n```\n\n"
            "Please provide your answer in the following format:\n"
            "- **Main Point 1**: Detail...\n"
            "- **Main Point 2**: Detail...\n"
            "- **Main Point 2**: Detail...\n"
            "- **Main Point 2**: Detail...\n"
            "- **Main Point 2**: Detail...\n"
        ),
    }
    
    messages.append(text_message)
    return [HumanMessage(content=messages)]

# 멀티모달 RAG 체인 생성 함수
def multi_modal_rag_chain(retriever):
    # LLM 정의
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=500)

    # MultiQueryRetriever 생성
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,  # 기존에 정의된 retriever를 사용
        llm=llm,
    )
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    # 멀티모달 LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=2048)

    # RAG 파이프라인
    chain = (
        {
            "context": multi_query_retriever | RunnableLambda(split_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
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
    
    # 문서 검색 및 리트리버 가져오기
    start_time = time.time()
    retriever, references = get_relevant_documents(question_type, question_category, question)
    retrieval_time = time.time() - start_time
    logger.debug(f"Document retrieval took {retrieval_time:.2f} seconds")

    if not retriever:
        logger.debug("No relevant documents found.")
        return JsonResponse({"error": "No relevant documents found"}, status=404)

    # RAG 체인을 사용하여 GPT 응답 생성
    try:
        # RAG 체인 생성
        chain_multimodal_rag = multi_modal_rag_chain(retriever)

        # 체인을 사용하여 응답 생성
        response = chain_multimodal_rag.invoke({"question": question})

        completion_time = time.time() - start_time
        logger.debug(f"RAG completion took {completion_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during RAG processing: {e}")
        return JsonResponse({"error": "An error occurred during processing"}, status=500)

    if not response:
        logger.debug("No relevant documents found or no response generated.")
        return JsonResponse({"error": "No relevant documents found"}, status=404)

    # 응답 처리
    answer = response.strip()

    # 참고 정보(예: 문서 링크 등)를 생성하는 부분
    references_response = [
        {
            "title": ref['title'],
            "link": f"http://127.0.0.1:8000/media/documents/{ref['title']}#page={ref['page']}"
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

    # 카테고리 이름을 영어로 변환
    category_mapping = {
        "모집요강": "admission_guideline",
        "입시결과": "admission_results",
        "기출문제": "past_exams",
        "대학생활": "campus_life",
        "면접/실기": "interview_practice"
    }

    categories = ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]
    retrievers = {}

    for category in categories:
        english_category = category_mapping.get(category, category)
        documents = model_class.objects.filter(category=category)
        if documents.exists():
            doc_contents = [(doc.title, doc.content, doc.page, doc.category) for doc in documents]
            embedding_function = OpenAIEmbeddings()
            vectorstore = Chroma(
                collection_name=f"{english_category}_collection",
                embedding_function=embedding_function,
                persist_directory=f"vectorDB/{english_category}_vectorDB"
            )
            retrievers[category] = create_multi_vector_retriever(vectorstore, doc_contents)

    if question_category:
        retriever = retrievers.get(question_category)
        if retriever:
            relevant_docs = retriever.vectorstore.similarity_search_with_score(question, k=max_docs)
        else:
            return "", []
    else:
        # 카테고리가 지정되지 않은 경우 모든 리트리버에서 검색
        relevant_docs = []
        for category, retriever in retrievers.items():
            category_docs = retriever.vectorstore.similarity_search_with_score(question, k=3)
            relevant_docs.extend(category_docs)

    if not relevant_docs:
        return "", []

    # 검색된 문서에서 메타데이터를 추출하여 context 및 references 생성
    references = []
    for doc in relevant_docs[:max_docs]:
        metadata = doc[0].metadata
        references.append({
            "title": metadata.get("title", "Unknown Title"),
            "page": metadata.get("page", "Unknown Page"),
            "category": metadata.get("category", "Unknown Category")
        })
    return retriever, references