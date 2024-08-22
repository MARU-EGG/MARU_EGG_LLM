import os
import sys
import platform
import django
from django.conf import settings

if platform.system() == "Darwin":
    project_root = '/Users/euntaek/Documents/MyProject/MARU_EGG_LLM'
elif platform.system() == "Linux":
    project_root = '/home/ubuntu/MARU_EGG_LLM'
else:
    raise EnvironmentError("알 수 없는 환경입니다. 적절한 경로를 설정하세요.")

sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.local')

django.setup()

from maruegg.tasks import process_files

def delete_processed_file(file_path):
    """
    파싱이 완료된 후 files 폴더의 파일을 삭제합니다.
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")

if __name__ == "__main__":
    process_files()
    files_dir = os.path.join(settings.MEDIA_ROOT, 'files')

    if os.path.exists(files_dir):
        for filename in os.listdir(files_dir):
            file_path = os.path.join(files_dir, filename)
            delete_processed_file(file_path)
    else:
        print(f"Directory not found: {files_dir}")