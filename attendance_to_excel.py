# Enhanced attendance_to_excel.py
import pandas as pd
import os
import datetime
import logging
import sqlite3
from typing import Tuple, Optional, List
from datetime import time as dt_time


class AttendanceExcelConverter:
    def __init__(self):
        self.attendance_dir = "my_templates"
        self.attendance_start = dt_time(7, 0)  # 7:00 AM
        self.attendance_end = dt_time(11, 0)  # 10:00 AM
        self.departure_start = dt_time(12, 0)  # 12:00 PM
        self.report_time = dt_time(11, 0)  # 3:00 PM
        os.makedirs(self.attendance_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)

    def is_within_time_window(self, current_time: datetime.time,
                              start_time: dt_time, end_time: Optional[dt_time] = None) -> bool:
        """Check if current time is within specified window"""
        if end_time:
            return start_time <= current_time <= end_time
        return current_time >= start_time

    def get_daily_excel_path(self, date_str: Optional[str] = None) -> str:
        """Get path for daily Excel file and ensure it exists"""
        date_str = date_str or datetime.datetime.now().strftime('%Y-%m-%d')
        excel_path = os.path.join(self.attendance_dir, f"attendance_{date_str}.xlsx")

        # Regenerate Excel file if it doesn't exist
        if not os.path.exists(excel_path):
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            pd.DataFrame(columns=["Student ID", "Name", "Attendance Time", "Departure Time", "Status"]) \
                .to_excel(excel_path, index=False)
            logging.info(f"Regenerated Excel attendance file for {date_str}")

        return excel_path

    def initialize_daily_sheet(self, date_str: str, student_list: List[str]) -> None:
        """Create new Excel sheet for the day with all students marked absent"""
        excel_path = self.get_daily_excel_path(date_str)

        if not os.path.exists(excel_path):
            df = pd.DataFrame({
                "Student ID": student_list,
                "Name": [""] * len(student_list),
                "Attendance Time": [""] * len(student_list),
                "Departure Time": [""] * len(student_list),
                "Status": ["Absent"] * len(student_list)
            })
            df.to_excel(excel_path, index=False)
            logging.info(f"Created new attendance sheet for {date_str}")

    def update_attendance_record(self, student_id: str, status: str) -> bool:
        """
        Update attendance record in Excel file with time window enforcement
        Returns True if update was successful, False otherwise
        """
        current_time = datetime.datetime.now().time()
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        excel_path = self.get_daily_excel_path(date_str)

        # Check time windows
        if status == "Present" and not self.is_within_time_window(current_time, self.attendance_start,
                                                                  self.attendance_end):
            logging.warning(f"Outside attendance window (7AM-10AM) for {student_id}")
            return False
        if status == "Left" and not self.is_within_time_window(current_time, self.departure_start):
            logging.warning(f"Outside departure window (after 12PM) for {student_id}")
            return False

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            mask = df["Student ID"] == student_id

            if mask.any():
                current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

                if status == "Present" and df.loc[mask, "Attendance Time"].iloc[0] == "":
                    df.loc[mask, "Attendance Time"] = current_time_str
                    df.loc[mask, "Status"] = "Present"
                    logging.info(f"Marked {student_id} as Present at {current_time_str}")
                elif status == "Left" and df.loc[mask, "Departure Time"].iloc[0] == "" and df.loc[mask, "Status"].iloc[
                    0] == "Present":
                    df.loc[mask, "Departure Time"] = current_time_str
                    logging.info(f"Marked {student_id} as Left at {current_time_str}")
                else:
                    logging.warning(f"Duplicate {status} attempt for {student_id}")
                    return False

                df.to_excel(excel_path, index=False)
                return True

        return False

    def generate_daily_report(self, date_str: Optional[str] = None) -> str:
        """Generate comprehensive daily report at 3PM automatically"""
        date_str = date_str or datetime.datetime.now().strftime('%Y-%m-%d')
        excel_path = self.get_daily_excel_path(date_str)

        if not os.path.exists(excel_path):
            return ""

        # Read attendance data from Excel
        df = pd.read_excel(excel_path)

        present = len(df[df['Status'] == 'Present'])
        absent = len(df) - present

        # Generate detailed report
        report_content = (
            f"Daily Report - {date_str}\n"
            f"Present: {present} | Absent: {absent}\n\n"
            f"Attendance Details:\n"
            f"{df.to_string(index=False)}"
        )

        report_path = os.path.join(self.attendance_dir, f"report_{date_str}.txt")
        with open(report_path, 'w') as f:
            f.write(report_content)

        logging.info(f"Generated daily report for {date_str}: Present {present} | Absent {absent}")
        return report_path

    def auto_generate_reports(self) -> None:
        """Check if it's report time and generate reports if needed"""
        if self.is_within_time_window(datetime.datetime.now().time(), self.report_time):
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')
            if not os.path.exists(os.path.join(self.attendance_dir, f"report_{date_str}.txt")):
                self.generate_daily_report(date_str)

    def convert_to_excel(self) -> None:
        """Convert database records to Excel (maintain original functionality)"""
        conn = sqlite3.connect('attendance.db')
        try:
            df = pd.read_sql('SELECT * FROM attendance', conn)
            if not df.empty:
                date_str = datetime.datetime.now().strftime('%Y-%m-%d')
                excel_path = self.get_daily_excel_path(date_str)
                df.to_excel(excel_path, index=False)
                logging.info(f"Converted database records to {excel_path}")
        finally:
            conn.close()


if __name__ == '__main__':
    converter = AttendanceExcelConverter()
