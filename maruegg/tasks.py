import os
import fitz
import camelot
import shutil
from .models import Document1, Document2, Document3
from django.conf import settings

# 카테고리 매핑
category_mapping = {
    "모집요강": "admission_guideline",
    "입시결과": "admission_results",
    "기출문제": "past_exams",
    "대학생활": "campus_life",
    "면접/실기": "interview_practice"
}

def parse_pdf_file(file_path, doc_type, doc_category):
    # 해당 문서에 맞는 모델 클래스 가져오기
    model_class = get_model_class(doc_type)
    
    # 벡터 DB 경로 삭제 (해당 문서에 대한 폴더 삭제)
    delete_vector_db_folder(doc_type, doc_category)
    
    # PDF 문서 열기
    pdf_document = fitz.open(file_path)
    title = os.path.basename(file_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text")
        page_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        page_text = ' '.join(page_lines)

        # PDF 페이지에서 테이블 추출
        tables = camelot.read_pdf(file_path, pages=str(page_num + 1), flavor='stream')
        if tables:
            for table in tables:
                table_text = table.df.to_csv(index=False)
                page_text += '\n' + table_text
        
        # 문서 저장
        model_class.objects.create(
            title=title,
            content=page_text,
            page=page_num + 1,
            category=doc_category
        )

def get_model_class(doc_type):
    if doc_type == "수시":
        return Document1
    elif doc_type == "정시":
        return Document2
    elif doc_type == "편입학":
        return Document3
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

def delete_vector_db_folder(doc_type, doc_category):
    """해당 문서에 대한 벡터 DB 폴더를 삭제하는 함수"""
    # 모델 클래스 이름 가져오기
    model_class = get_model_class(doc_type)
    class_name = model_class.__name__

    # 카테고리 이름을 영어로 변환
    english_category = category_mapping.get(doc_category, doc_category)

    # 벡터 DB 경로 설정
    persist_directory = os.path.join(settings.BASE_DIR, f"vectorDB/{class_name}/{english_category}_vectorDB")

    # 폴더가 존재하는 경우 삭제
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted vector store at {persist_directory}")
    else:
        print(f"Vector store path {persist_directory} does not exist, no need to delete.")

def process_files():
    files_dir = os.path.join(settings.MEDIA_ROOT, 'files')
    for filename in os.listdir(files_dir):
        if filename.endswith('.pdf'):
            try:
                doc_type, doc_category_with_ext = filename.split('_', 1)
                doc_category = doc_category_with_ext.rsplit('.', 1)[0]
                file_path = os.path.join(files_dir, filename)
                parse_pdf_file(file_path, doc_type, doc_category)
            except ValueError as e:
                print(f"Error processing file {filename}: {e}")
                continue
