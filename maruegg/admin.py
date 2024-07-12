from django.contrib import admin
from .models import Document1, Document2, Document3

@admin.action(description='Delete all documents')
def delete_all_documents(modeladmin, request, queryset):
    Document1.objects.all().delete()

from django.contrib import admin
from .models import Document1, Document2, Document3

class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'part', 'category', 'created_at')
    list_filter = ('category',)

admin.site.register(Document1, DocumentAdmin)
admin.site.register(Document2, DocumentAdmin)
admin.site.register(Document3, DocumentAdmin)

