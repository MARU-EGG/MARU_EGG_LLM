from django.db import models

class Document1(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('대학생활', '대학생활'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    part = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.part})"
    
class Document2(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('대학생활', '대학생활'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    part = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.part})"

class Document3(models.Model):
    CATEGORY_CHOICES = [
        ('모집요강', '모집요강'),
        ('입시결과', '입시결과'),
        ('기출문제', '기출문제'),
        ('대학생활', '대학생활'),
        ('면접/실기', '면접/실기'),
    ]
    title = models.CharField(max_length=200)
    content = models.TextField()
    part = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    def __str__(self):
        return f"{self.title} (Part {self.part})"
