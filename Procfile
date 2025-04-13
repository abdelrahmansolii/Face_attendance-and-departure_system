# Core services (must run first)
face_registration: python face_registration_api.py
attendance: python attendance_api.py

# Web interface (main access point)
web: gunicorn app:app --bind=0.0.0.0:$PORT --workers=2

# Data export (optional)
data_export: python data_fetch_csv_api.py