from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []
CSRF_TRUSTED_ORIGINS = ['http://localhost', 'http://127.0.0.1']

DJANGO_APPS+=[]
PROJECT_APPS+=[]
THIRD_PARTY_APPS+=[] # 추후에 디버그 툴바 넣어줘야 함


INSTALLED_APPS = DJANGO_APPS + PROJECT_APPS + THIRD_PARTY_APPS

# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATICFILES_DIRS = [
    BASE_DIR / 'static'
]