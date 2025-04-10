import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import csv
from datetime import time as dt_time

os.chdir(r"E:\Face_recognition\Face_recognition_attendance")

# Initialize detector, predictor, and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Recognizer:
    def __init__(self):
        self.attendance_csv = os.path.join("E:\\Face_recognition\\Face_recognition_attendance\\my_templates",
                                         "attendance.csv")
        self.initialize_attendance_csv()

        # Time window configuration
        self.attendance_start = dt_time(7, 0)  # 7:00 AM
        self.attendance_end = dt_time(10, 0)   # 10:00 AM
        self.departure_start = dt_time(12, 0)  # 12:00 PM
        self.report_time = dt_time(14,0)    # 2:00 PM

        self.font = cv2.FONT_ITALIC
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.frame_cnt = 0
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    def initialize_attendance_csv(self):
        os.makedirs(os.path.dirname(self.attendance_csv), exist_ok=True)
        if not os.path.isfile(self.attendance_csv):
            with open(self.attendance_csv, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Timestamp", "Status"])

    def get_daily_attendance_file(self):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_file = os.path.join("E:\\Face_recognition\\Face_recognition_attendance\\my_templates",
                                  f"attendance_{current_date}.csv")

        # Check if file exists, if not create it
        if not os.path.exists(daily_file):
            os.makedirs(os.path.dirname(daily_file), exist_ok=True)
            with open(daily_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Timestamp", "Status"])
            logging.info(f"Created new daily attendance file: {daily_file}")

        return daily_file

    def is_within_time_window(self, start_time, end_time=None):
        """Check if current time is within specified window"""
        current_time = datetime.datetime.now().time()
        if end_time:
            return start_time <= current_time <= end_time
        return current_time >= start_time

    def log_attendance(self, name):
        """Enhanced attendance logging with time windows and custom messages"""
        current_datetime = datetime.datetime.now()
        current_time = current_datetime.time()
        current_date_str = current_datetime.strftime("%Y-%m-%d")
        current_time_str = current_datetime.strftime("%H:%M:%S")

        attendance_file = self.get_daily_attendance_file()

        # Initialize daily file if doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Timestamp", "Status"])

        # Read existing data
        try:
            df = pd.read_csv(attendance_file)
        except:
            df = pd.DataFrame(columns=["Name", "Timestamp", "Status"])

        # Check for existing entries today
        existing_entries = df[(df["Name"] == name) &
                              (df["Timestamp"].str.startswith(current_date_str))]

        # Late attendance case (after 10:00 AM but before 12:00 PM)
        if (current_time > self.attendance_end and
                current_time < self.departure_start and
                "Present" not in existing_entries["Status"].values):
            message = f"{name}, you are absent because of latency. Please go to student affairs office."
            print(message)  # Console output
            return message  # Will be displayed in camera window

        # Normal attendance period (7:00-10:00 AM)
        if self.is_within_time_window(self.attendance_start, self.attendance_end):
            if "Present" not in existing_entries["Status"].values:
                new_entry = {
                    "Name": name,
                    "Timestamp": f"{current_date_str} {current_time_str}",
                    "Status": "Present"
                }
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                return f"{name} marked Present at {current_time_str}"
            else:
                present_time = existing_entries[existing_entries["Status"] == "Present"]["Timestamp"].values[0]
                return f"{name} already marked Present at {present_time.split()[1]}"

        # Departure period (after 12:00 PM)
        elif self.is_within_time_window(self.departure_start):
            if "Present" in existing_entries["Status"].values and "Left" not in existing_entries["Status"].values:
                new_entry = {
                    "Name": name,
                    "Timestamp": f"{current_date_str} {current_time_str}",
                    "Status": "Left"
                }
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                return f"{name} marked Left at {current_time_str}"
            elif "Left" in existing_entries["Status"].values:
                left_time = existing_entries[existing_entries["Status"] == "Left"]["Timestamp"].values[0]
                return f"{name} already departed at {left_time.split()[1]}"

        # Generate report if it's report time (3:00 PM)
        if self.is_within_time_window(self.report_time):
            self.generate_daily_report()

    def generate_daily_report(self):
        """Generate detailed daily summary report with names and times"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        attendance_file = self.get_daily_attendance_file()
        report_file = os.path.join("E:\\Face_recognition\\Face_recognition_attendance\\my_templates",
                                   f"daily_report_{current_date}.csv")

        if os.path.exists(attendance_file):
            # Read attendance data
            df = pd.read_csv(attendance_file)

            # Get present students with their times
            present_df = df[df["Status"] == "Present"]
            present_details = []
            for _, row in present_df.iterrows():
                name = row["Name"]
                arrival_time = row["Timestamp"].split()[1]  # Get time part only
                departure_record = df[(df["Name"] == name) & (df["Status"] == "Left")]
                departure_time = departure_record["Timestamp"].values[0].split()[
                    1] if not departure_record.empty else "Not departed"
                present_details.append(f"{name}: Arrived at {arrival_time}, Departed at {departure_time}")

            # Get absent students (in database but not marked present)
            absent_students = [name for name in self.face_name_known_list
                               if name not in present_df["Name"].values]

            # Prepare report content for CSV
            report_data = {
                "Report_Type": [],
                "Details": []
            }

            # Add header
            report_data["Report_Type"].append("Header")
            report_data["Details"].append(f"Daily Attendance Report - {current_date}")

            # Add present students
            for detail in present_details:
                report_data["Report_Type"].append("Present")
                report_data["Details"].append(detail)

            # Add present count
            report_data["Report_Type"].append("Present_Count")
            report_data["Details"].append(f"Total Present: {len(present_details)}")

            # Add absent header
            report_data["Report_Type"].append("Absent_Header")
            report_data["Details"].append("=== Absent Students ===")

            # Add absent students
            for student in absent_students:
                report_data["Report_Type"].append("Absent")
                report_data["Details"].append(student)

            # Add absent count
            report_data["Report_Type"].append("Absent_Count")
            report_data["Details"].append(f"Total Absent: {len(absent_students)}")

            # Save to CSV
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(report_file, index=False)

            # Also create a human-readable text file
            text_report_file = os.path.join("E:\\Face_recognition\\Face_recognition_attendance\\my_templates",
                                            f"daily_report_{current_date}.txt")
            with open(text_report_file, 'w') as f:
                f.write(f"Daily Attendance Report - {current_date}\n")
                f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("=== Present Students ===\n")
                f.write("\n".join(present_details) + "\n")
                f.write(f"\nTotal Present: {len(present_details)}\n\n")
                f.write("=== Absent Students ===\n")
                f.write("\n".join(absent_students) + "\n")
                f.write(f"\nTotal Absent: {len(absent_students)}\n")

            logging.info(f"Generated detailed daily report for {current_date}")

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)
            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                img_rd = cv2.flip(img_rd, 1)
                kk = cv2.waitKey(1)

                faces = detector(img_rd, 0)
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                  tuple([d.left(), d.top()]),
                                                  tuple([d.right(), d.bottom()]),
                                                  (255, 255, 255), 2)

                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)
                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    if self.current_frame_face_cnt == 0:
                        logging.debug("No faces in this frame!!!")
                        self.current_frame_face_name_list = []
                    else:
                        logging.debug("scene 2.2 Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        for k in range(len(faces)):
                            logging.debug("For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                recognized_name = self.face_name_known_list[similar_person_num]
                                self.current_frame_face_name_list[k] = recognized_name
                                logging.debug("Face recognition result: %s", recognized_name)
                                message = self.log_attendance(recognized_name)

                                # Display message below the face
                                text_position = (faces[k].left(), faces[k].bottom() + 30)
                                cv2.putText(img_rd, message, text_position,
                                            self.font, 0.7, (0, 0, 255), 2)  # Red text

                                # Also show the name above the face (original behavior)
                                cv2.putText(img_rd, recognized_name,
                                            (faces[k].left(), faces[k].top() - 10),
                                            self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                            else:
                                logging.debug("Face recognition result: Unknown person")

                        self.draw_note(img_rd)

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)
                logging.debug("Frame ends\n\n")

    def run(self):
        """Main method to start the face recognition process"""
        cap = cv2.VideoCapture(0)
        try:
            self.process(cap)
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    recognizer = Face_Recognizer()
    recognizer.run()

if __name__ == '__main__':
    main()