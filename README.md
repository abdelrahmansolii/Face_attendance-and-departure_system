## API Documentation

### 1. Face Registration API (Port 5000) face_registration_api.py
- **Base URL**: `http://localhost:5000`
- Endpoints:
  - `POST /register/start` - Initialize registration session
  - `GET /register/capture` - Capture current frame
  - `POST /register/save` - Save detected face with name
  - `POST /register/clear` - Clear all registered faces

### 2. Attendance Verification API (Port 5001) attendance_api.py

- **Base URL**: `http://localhost:5001`
- Endpoints:
  - `POST /attendance/verify` - Verify face and mark attendance
  - `GET /attendance/dates` - List available attendance dates
  - `GET /attendance/export` - Export attendance data (CSV)
  - `GET /attendance/report` - Generate attendance report

### 3. Data Fetch API (Port 5002)  data_fetch_csv_api.py
- **Base URL**: `http://localhost:5002`
- Endpoints:
  - `GET /` - View today's attendance
  - `POST /view_attendance` - View attendance for selected date
  - `GET /export_attendance` - Export attendance data

### 4. Web Interface (Port 5003) app.py
- **Base URL**: `http://localhost:5003`
- Routes:
  - `/` - Main dashboard
  - `/view_attendance` - Date-filtered attendance view