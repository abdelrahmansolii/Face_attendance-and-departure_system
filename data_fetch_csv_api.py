from flask import Flask, render_template, request, redirect, url_for, flash, send_file

import pandas as pd

import os

from datetime import datetime, date

app = Flask(__name__, template_folder='my_templates')

app.secret_key = 'FLASK_SECRET_KEY'

ATTENDANCE_DIR = "E:\\Face_recognition\\Face_recognition_attendance\\my_templates"

def get_available_dates() -> list:
    """Get all dates with attendance records from Excel files"""

    dates = []

    for filename in os.listdir(ATTENDANCE_DIR):
        if filename.startswith("attendance_") and filename.endswith(".csv"):
            date_str = filename[11:-4]  # Extract date from "attendance_YYYY-MM-DD.csv"

            try:

                datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_str)

            except ValueError:
                continue

    return sorted(dates, reverse=True)

@app.route('/')

def index():

    """Main page showing today's attendance"""
    today = date.today().strftime('%Y-%m-%d')

    available_dates = get_available_dates()
    # Try to load today's data
    file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        attendance_data = df.to_dict('records')

    else:

        attendance_data = []

    return render_template('index.html',

                         selected_date=today,

                         available_dates=available_dates,

                         attendance_data=attendance_data,

                         no_data=not bool(attendance_data),

                         date=date)  # Pass date object to template


@app.route('/view_attendance', methods=['POST'])

def view_attendance():
    """View attendance for selected date"""

    selected_date = request.form.get('selected_date')

    try:

        # Validate date format
        datetime.strptime(selected_date, '%Y-%m-%d')

    except ValueError:

        flash('Invalid date format', 'error')

        return redirect(url_for('index'))

    file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{selected_date}.csv")


    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        attendance_data = df.to_dict('records')

    else:

        attendance_data = []


    return render_template('index.html',

                         selected_date=selected_date,

                         available_dates=get_available_dates(),

                         attendance_data=attendance_data,

                         no_data=not bool(attendance_data),

                         date=date)  # Pass date object to template


@app.route('/export_attendance')

def export_attendance():

    """Export attendance data to Excel"""

    selected_date = request.args.get('date', date.today().strftime('%Y-%m-%d'))



    file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{selected_date}.csv")


    if os.path.exists(file_path):

        return send_file(

            file_path,

            as_attachment=True,
            download_name=f"attendance_{selected_date}.csv",
            mimetype='text/csv'

        )

    else:
        flash('No attendance data available for selected date', 'warning')

        return redirect(url_for('index'))

if __name__ == '__main__':



    app.run(host='0.0.0.0', port=5002, debug=True)