from django.http import JsonResponse
from ..models import Prompt
from django.views.decorators.csrf import csrf_exempt
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser

# 프롬프트 생성
@swagger_auto_schema(
    method='post',
    operation_description="프롬프트를 생성하는 API",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'question_type': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Type of the question (수시, 정시, 편입학)',
                enum=['수시', '정시', '편입학']
            ),
            'question_category': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
                enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
            ),
            'prompt_text': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Prompt text content (여러 줄 입력 가능)'
            ),
        },
        required=['question_type', 'question_category', 'prompt_text']
    ),
    responses={201: '프롬프트가 성공적으로 생성되었습니다.', 400: '잘못된 요청', 500: '서버 오류'}
)
@csrf_exempt
@api_view(['POST'])
def create_prompt(request):
    if request.method == "POST":
        data = request.data  # 수정된 부분: request.body 대신 request.data 사용
        question_type = data.get("question_type")
        question_category = data.get("question_category")
        prompt_text = data.get("prompt_text")

        if not question_type or not question_category or not prompt_text:
            return JsonResponse({"error": "모든 필드를 입력해야 합니다."}, status=400)

        try:
            prompt, created = Prompt.objects.get_or_create(
                question_type=question_type,
                question_category=question_category,
                defaults={"prompt_text": prompt_text}
            )
            if not created:
                return JsonResponse({"error": "이미 존재하는 프롬프트입니다."}, status=400)

            return JsonResponse({"message": "프롬프트가 성공적으로 생성되었습니다."}, status=201)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)



@swagger_auto_schema(
    method='get',
    operation_description="프롬프트를 조회하는 API (특정 type과 category에 대해 단일 프롬프트 반환)",
    manual_parameters=[
        openapi.Parameter(
            'type',
            openapi.IN_QUERY,
            description='Type of the question (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: "Success", 400: "Invalid request", 404: "No prompt found"}
)
@csrf_exempt
@api_view(['GET'])
def get_prompt(request):
    question_type = request.GET.get("type")
    question_category = request.GET.get("category")

    # 유효성 검사
    if not question_type or question_type not in ["수시", "정시", "편입학"]:
        return JsonResponse({"error": "Invalid or missing type provided"}, status=400)
    
    if not question_category or question_category not in ["모집요강", "입시결과", "기출문제", "대학생활", "면접/실기"]:
        return JsonResponse({"error": "Invalid or missing category provided"}, status=400)

    try:
        # 단일 프롬프트 조회
        prompt = Prompt.objects.get(
            question_type=question_type,
            question_category=question_category
        )
        return JsonResponse({"prompt_text": prompt.prompt_text}, status=200)
    except Prompt.DoesNotExist:
        return JsonResponse({"error": "프롬프트를 찾을 수 없습니다."}, status=404)


# 프롬프트 수정
@swagger_auto_schema(
    method='put',
    operation_description="프롬프트를 업데이트하는 API",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'question_type': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Type of the question (수시, 정시, 편입학)',
                enum=['수시', '정시', '편입학']
            ),
            'question_category': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
                enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
            ),
            'prompt_text': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Updated prompt text content (여러 줄 입력 가능)'
            ),
        },
        required=['question_type', 'question_category', 'prompt_text']
    ),
    responses={200: '프롬프트가 성공적으로 업데이트되었습니다.', 400: '잘못된 요청', 404: '프롬프트를 찾을 수 없습니다.'}
)
@csrf_exempt
@api_view(['PUT'])
def update_prompt(request):
    data = request.data  # 수정된 부분: request.body 대신 request.data 사용
    question_type = data.get("question_type")
    question_category = data.get("question_category")
    prompt_text = data.get("prompt_text")

    if not prompt_text:
        return JsonResponse({"error": "프롬프트 텍스트를 입력해야 합니다."}, status=400)

    try:
        prompt = Prompt.objects.get(
            question_type=question_type,
            question_category=question_category
        )
        prompt.prompt_text = prompt_text
        prompt.save()
        return JsonResponse({"message": "프롬프트가 성공적으로 업데이트되었습니다."}, status=200)
    except Prompt.DoesNotExist:
        return JsonResponse({"error": "프롬프트를 찾을 수 없습니다."}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# 프롬프트 삭제
@swagger_auto_schema(
    method='delete',
    operation_description="프롬프트를 삭제하는 API",
    manual_parameters=[
        openapi.Parameter(
            'question_type',
            openapi.IN_QUERY,
            description='Type of the question (수시, 정시, 편입학)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['수시', '정시', '편입학']
        ),
        openapi.Parameter(
            'question_category',
            openapi.IN_QUERY,
            description='Category of the question (모집요강, 입시결과, 기출문제, 대학생활, 면접/실기)',
            required=True,
            type=openapi.TYPE_STRING,
            enum=['모집요강', '입시결과', '기출문제', '대학생활', '면접/실기']
        ),
    ],
    responses={200: '프롬프트가 성공적으로 삭제되었습니다.', 404: '프롬프트를 찾을 수 없습니다.'}
)
@csrf_exempt
@api_view(['DELETE'])
def delete_prompt(request):
    question_type = request.GET.get("question_type")
    question_category = request.GET.get("question_category")

    try:
        prompt = Prompt.objects.get(
            question_type=question_type,
            question_category=question_category
        )
        prompt.delete()
        return JsonResponse({"message": "프롬프트가 성공적으로 삭제되었습니다."}, status=200)
    except Prompt.DoesNotExist:
        return JsonResponse({"error": "프롬프트를 찾을 수 없습니다."}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
