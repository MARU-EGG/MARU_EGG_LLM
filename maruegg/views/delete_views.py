from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from ..models import Document1, Document2, Document3

@swagger_auto_schema(
    method='delete',
    operation_description="DB내 데이터를 삭제합니다. (특정 type, category별 또는 전체)",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the documents to delete (수시, 정시, 편입학)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the documents to delete (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=False,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request"}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_documents(request):
    doc_type = request.GET.get("type")
    doc_category = request.GET.get("category")
    
    model_classes = {
        "수시": Document1,
        "정시": Document2,
        "편입학": Document3
    }
    
    if doc_type and doc_type not in model_classes:
        return JsonResponse({"error": "Invalid type provided"}, status=400)
    
    if doc_category and doc_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
        return JsonResponse({"error": "Invalid category provided"}, status=400)
    
    if doc_type:
        model_class = model_classes[doc_type]
        queryset = model_class.objects.all()
        if doc_category:
            queryset = queryset.filter(category=doc_category)
        queryset.delete()
    else:
        for model_class in model_classes.values():
            queryset = model_class.objects.all()
            if doc_category:
                queryset = queryset.filter(category=doc_category)
            queryset.delete()
    
    return JsonResponse({"message": "Documents deleted successfully"}, status=200)
