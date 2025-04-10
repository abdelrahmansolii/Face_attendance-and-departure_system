import os

class Config:
    UPLOAD_FOLDER = os.path.abspath('data/faces')
    ATTENDANCE_FOLDER = os.path.abspath('data/attendance')
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev_key_123')