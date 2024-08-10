import os
import fitz
import camelot
from .models import Document1, Document2, Document3
from django.conf import settings

def parse_pdf_file(file_path, doc_type, doc_category):
    model_class = get_model_class(doc_type)
    
    pdf_document = fitz.open(file_path)
    title = os.path.basename(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text")
        page_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        page_text = ' '.join(page_lines)

        tables = camelot.read_pdf(file_path, pages=str(page_num+1), flavor='stream')
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

def get_model_class(doc_type):
    if doc_type == "수시":
        return Document1
    elif doc_type == "정시":
        return Document2
    elif doc_type == "편입학":
        return Document3
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

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
