from django.contrib import admin
from .models import Document1, Document2, Document3, Prompt

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

admin.site.register(Document1, DocumentAdmin)
admin.site.register(Document2, DocumentAdmin)
admin.site.register(Document3, DocumentAdmin)
admin.site.register(Prompt, PromptAdmin)
