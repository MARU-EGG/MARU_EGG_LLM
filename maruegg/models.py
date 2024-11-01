from django.db import models

class Document1(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    page = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.page})"
    
class Document2(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    page = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.page})"

class Document3(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    page = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.page})"

class Prompt(models.Model):
    question_type = models.CharField(max_length=50)
    question_category = models.CharField(max_length=50)
    prompt_text = models.TextField()

    class Meta:
        unique_together = ('question_type', 'question_category')
        verbose_name = "Prompt"
        verbose_name_plural = "Prompts"

    def __str__(self):
        return f"{self.question_type} - {self.question_category}"