from django.shortcuts import render, redirect

def main(request):
    return render(request, "maruegg/upload_html_file.html")