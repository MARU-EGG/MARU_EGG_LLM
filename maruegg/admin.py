from django.contrib import admin
from .models import Document

@admin.action(description='Delete all documents')
def delete_all_documents(modeladmin, request, queryset):
    Document.objects.all().delete()

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'part', 'created_at')
    actions = [delete_all_documents]
