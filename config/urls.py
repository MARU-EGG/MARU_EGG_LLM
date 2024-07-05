from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("maruegg", include("maruegg.urls")),
    path("accounts", include("accounts.urls")),
    path("", TemplateView.as_view(template_name="base.html"), name="root"),
]
