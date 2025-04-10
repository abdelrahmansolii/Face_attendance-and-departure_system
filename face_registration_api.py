import os
import time
import logging
import shutil
import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('registration.log'),
        logging.StreamHandler()
    ]
)

# Global state for camera (initialized when needed)
camera = None
current_frame = None


def get_existing_faces_count():
    """Helper function to count existing face directories"""
    if os.path.exists(Config.UPLOAD_FOLDER):
        return len([name for name in os.listdir(Config.UPLOAD_FOLDER)
                    if os.path.isdir(os.path.join(Config.UPLOAD_FOLDER, name))])
    return 0


@app.route('/')
def index():
    return "Face Registration API is running! (Port 5000)"


@app.route('/register/start', methods=['POST'])
def start_registration():
    """Initialize registration session"""
    try:
        count = get_existing_faces_count()
        return jsonify({
            'status': 'success',
            'message': 'Registration session ready',
            'existing_faces': count
        }), 200
    except Exception as e:
        logging.error(f"Start error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/register/capture', methods=['GET'])
def capture_frame():
    """Capture and return current frame with faces"""
    global camera, current_frame

    try:
        # Initialize camera if not already open
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(2)  # Allow camera to warm up

        ret, frame = camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        frame = cv2.flip(frame, 1)
        faces = detector(frame, 0)
        current_frame = frame

        # Convert to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        return Response(
            jpeg.tobytes(),
            mimetype='image/jpeg',
            headers={
                'face-count': str(len(faces)),
                'existing-faces': str(get_existing_faces_count())
            }
        )
    except Exception as e:
        logging.error(f"Capture error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # Don't release camera here to allow for quick subsequent captures
        pass


@app.route('/register/save', methods=['POST'])
def save_face():
    """Save detected face with provided name"""
    global current_frame

    try:
        if current_frame is None:
            raise ValueError("No frame available. Capture first.")

        name = request.form.get('name', '').strip()
        if not name:
            raise ValueError("Name is required")

        faces = detector(current_frame, 0)
        if len(faces) != 1:
            raise ValueError("Frame must contain exactly one face")

        # Create directory for new person
        person_id = get_existing_faces_count() + 1
        person_dir = os.path.join(
            Config.UPLOAD_FOLDER,
            f"person_{person_id}_{secure_filename(name)}"
        )
        os.makedirs(person_dir, exist_ok=True)

        # Crop and save face
        face = faces[0]
        face_img = current_frame[face.top():face.bottom(), face.left():face.right()]

        # Resize to standard size if needed
        face_img = cv2.resize(face_img, (256, 256)) if face_img.size > 0 else None

        if face_img is None or face_img.size == 0:
            raise ValueError("Failed to extract face region")

        timestamp = int(time.time())
        save_path = os.path.join(person_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(save_path, face_img)

        logging.info(f"Saved face for {name} at {save_path}")

        return jsonify({
            'status': 'success',
            'path': save_path,
            'person_id': person_id,
            'name': name,
            'timestamp': timestamp
        }), 200

    except Exception as e:
        logging.error(f"Save error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/register/clear', methods=['POST'])
def clear_data():
    """Clear all registered faces"""
    try:
        if os.path.exists(Config.UPLOAD_FOLDER):
            shutil.rmtree(Config.UPLOAD_FOLDER)
            os.makedirs(Config.UPLOAD_FOLDER)

        logging.info("Cleared all face registration data")

        return jsonify({
            'status': 'success',
            'message': 'All face data cleared',
            'remaining_faces': 0
        }), 200
    except Exception as e:
        logging.error(f"Clear error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.teardown_appcontext
def release_camera(exception=None):
    """Ensure camera is released when app context tears down"""
    global camera
    if camera is not None:
        camera.release()
        camera = None


if __name__ == '__main__':
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)