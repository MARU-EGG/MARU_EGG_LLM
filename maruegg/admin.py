from django.contrib import admin
from .models import Document1, Document2, Document3, Prompt, TableOfContents

@admin.action(description='Delete all documents')
def delete_all_documents(modeladmin, request, queryset):
    Document1.objects.all().delete()

class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'page', 'category', 'created_at')
    list_filter = ('category',)

class PromptAdmin(admin.ModelAdmin):
    list_display = ('question_type', 'question_category', 'prompt_text')
    list_filter = ('question_type', 'question_category')
    search_fields = ('prompt_text',)

class TableOfContentsAdmin(admin.ModelAdmin):
    list_display = ('toc_type', 'toc_category', 'get_toc_preview')
    list_filter = ('toc_type', 'toc_category')
    search_fields = ('toc_text',)
    
    def get_toc_preview(self, obj):
        """목차 텍스트의 미리보기를 제공"""
        if len(obj.toc_text) > 50:
            return obj.toc_text[:50] + "..."
        return obj.toc_text
    get_toc_preview.short_description = 'TOC Preview'

admin.site.register(Document1, DocumentAdmin)
admin.site.register(Document2, DocumentAdmin)
admin.site.register(Document3, DocumentAdmin)
admin.site.register(Prompt, PromptAdmin)
admin.site.register(TableOfContents, TableOfContentsAdmin)
