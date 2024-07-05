from django.shortcuts import render
from django.http import HttpResponse
from PyPDF2 import PdfReader
import io
from bs4 import BeautifulSoup
import tempfile
import os

def main(request):
    return render(request, "maruegg/upload_html_file.html")

def upload_pdf(request):
    if request.method == "POST" and request.FILES.get("pdf"):
        pdf_file = request.FILES["pdf"]
        pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
        pdf_text = ""

        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        return render(request, "base.html", {"pdf_text": pdf_text})

    return HttpResponse("Invalid request", status=400)

def upload_html(request):
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