from django.contrib import admin
from django.urls import include, path, re_path
from django.views.generic import TemplateView
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.auth.decorators import login_required
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Swagger 설정
schema_view = get_schema_view(
    openapi.Info(
        title="MARU_EGG_LLM",
        default_version='ver 2.0',
        description="MARU_EGG_LLM 프로젝트 LLM 모델 사용을 위한 API",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="이메일"),
        license=openapi.License(name="mit"),
    ),
    public=False,
    permission_classes=[permissions.IsAuthenticated],
    url='https://marueggllmserver.com' if not settings.DEBUG else None
)

settings.LOGIN_URL = '/admin/login/'

urlpatterns = [
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', login_required(schema_view.without_ui(cache_timeout=0)), name='schema-json'),
    path(r'swagger', login_required(schema_view.with_ui('swagger', cache_timeout=0)), name='schema-swagger-ui'),
    path(r'redoc', login_required(schema_view.with_ui('redoc', cache_timeout=0)), name='schema-redoc-v1'),
    # main
    path("admin/", admin.site.urls),
    path("maruegg/", include("maruegg.urls")),
    path("accounts/", include("accounts.urls")),
    path("api-auth/", include("rest_framework.urls")),
    path("", TemplateView.as_view(template_name="base.html"), name="root"),
]

# 미디어 파일 설정
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
