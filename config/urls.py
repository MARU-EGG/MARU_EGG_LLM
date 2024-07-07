from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("admin/", admin.site.urls),
    path("maruegg/", include("maruegg.urls")),
    path("accounts/", include("accounts.urls")),
    path("", TemplateView.as_view(template_name="base.html"), name="root"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
