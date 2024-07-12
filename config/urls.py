from django.contrib import admin
from django.urls import include, path, re_path
from django.views.generic import TemplateView
from django.conf.urls.static import static
from django.conf import settings
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg       import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="MARU_EGG_LLM",
        default_version='ver 2.0',
        description="MARU_EGG_LLM 프로젝트 LLM 모델 사용을 위한 API",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="이메일"),
        license=openapi.License(name="mit"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path(r'swagger', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path(r'redoc', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc-v1'),
    # main
    path("admin/", admin.site.urls),
    path("maruegg/", include("maruegg.urls")),
    path("accounts/", include("accounts.urls")),
    path("", TemplateView.as_view(template_name="base.html"), name="root"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
