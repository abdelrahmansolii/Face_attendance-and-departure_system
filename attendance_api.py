import os

import datetime

import csv

import numpy as np

from flask import Flask, request, jsonify , Response

import cv2

import dlib



app = Flask(__name__)



# Initialize face recognition components

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")



# Configuration

ATTENDANCE_FOLDER = 'data/attendance_records/'

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)



# Load known faces

known_face_names = []

known_face_features = []





def load_known_faces():

    global known_face_names, known_face_features

    features_file = 'data/features_all.csv'

    if os.path.exists(features_file):

        with open(features_file, 'r') as f:

            for line in csv.reader(f):

                known_face_names.append(line[0])

                known_face_features.append([float(x) for x in line[1:129]])





load_known_faces()





@app.route('/')

def index():

    return "Attendance Verification API is running!"



@app.route('/test')

def test():

    return jsonify({"status": "active"})



@app.route('/attendance/verify', methods=['POST'])

def verify_face():

    """Process image and mark attendance"""

    try:

        if 'image' not in request.files:

            return jsonify({'error': 'No image provided'}), 400



        file = request.files['image']

        if file.filename == '':

            return jsonify({'error': 'Empty filename'}), 400



        # Process image

        img_bytes = file.read()

        img_np = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        img = cv2.flip(img, 1)



        # Detect faces

        faces = detector(img, 0)

        if not faces:

            return jsonify({'message': 'No faces detected'}), 200



        results = []

        current_time = datetime.datetime.now()



        for face in faces:

            # Get face features

            shape = predictor(img, face)

            face_features = face_reco_model.compute_face_descriptor(img, shape)



            # Find best match

            distances = [np.linalg.norm(np.array(face_features) - np.array(known))

                         for known in known_face_features]

            min_distance = min(distances) if distances else 1.0

            best_match = distances.index(min_distance) if min_distance < 0.4 else -1



            if best_match != -1:

                name = known_face_names[best_match]

                # Record attendance

                date_str = current_time.strftime("%Y-%m-%d")

                time_str = current_time.strftime("%H:%M:%S")

                record_file = os.path.join(ATTENDANCE_FOLDER, f"attendance_{date_str}.csv")



                with open(record_file, 'a', newline='') as f:

                    writer = csv.writer(f)

                    writer.writerow([name, time_str, "Present"])



                results.append({

                    'name': name,

                    'status': 'Present',

                    'time': time_str,

                    'confidence': float(1 - min_distance)

                })

            else:

                results.append({

                    'name': 'Unknown',

                    'status': 'Not recognized',

                    'confidence': float(1 - min_distance)

                })



        return jsonify({

            'faces_detected': len(faces),

            'recognitions': results

        }), 200



    except Exception as e:

        return jsonify({'error': str(e)}), 500





@app.route('/attendance/dates')

def get_available_dates():

    """List all dates with attendance records"""

    dates = []

    for filename in os.listdir(ATTENDANCE_FOLDER):

        if filename.startswith("attendance_") and filename.endswith(".csv"):

            date_str = filename[11:-4]

            try:

                datetime.strptime(date_str, '%Y-%m-%d')

                dates.append(date_str)

            except ValueError:

                continue

    return jsonify({'dates': sorted(dates, reverse=True)})





@app.route('/attendance/export')

def export_attendance():

    """Raw CSV export endpoint"""

    date_str = request.args.get('date')

    file_path = os.path.join(ATTENDANCE_FOLDER, f"attendance_{date_str}.csv")



    if os.path.exists(file_path):

        with open(file_path, 'r') as f:

            return Response(

                f.read(),

                mimetype='text/csv',

                headers={'Content-Disposition': f'attachment; filename=attendance_{date_str}.csv'}

            )

    return jsonify({'error': 'File not found'}), 404



@app.route('/attendance/report', methods=['GET'])

def generate_report():

    """Generate today's attendance report"""

    try:

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        report_file = os.path.join(ATTENDANCE_FOLDER, f"attendance_{date_str}.csv")



        if not os.path.exists(report_file):

            return jsonify({'message': 'No attendance records today'}), 404



        present = set()

        with open(report_file, 'r') as f:

            reader = csv.reader(f)

            for row in reader:

                if row:  # Skip empty lines

                    present.add(row[0])



        absent = [name for name in known_face_names if name not in present]



        return jsonify({

            'date': date_str,

            'present': list(present),

            'absent': absent,

            'total_present': len(present),

            'total_absent': len(absent)

        }), 200



    except Exception as e:

        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5001, debug=True)