from flask import Flask, render_template, request, redirect, url_for, flash, Response
import requests
from datetime import datetime, date
import os
import logging

app = Flask(__name__, template_folder='my_templates')
app.secret_key = 'FLASK_SECRET_KEY'

# Configuration
ATTENDANCE_API_URL = "http://localhost:5001"
API_TIMEOUT = 5

def get_available_dates():
    try:
        response = requests.get(f"{ATTENDANCE_API_URL}/attendance/dates", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return sorted(response.json().get('dates', []), reverse=True)
    except requests.exceptions.RequestException as e:
        logging.error(f"API Error: {str(e)}")
    return []

@app.route('/')
def index():
    try:
        # Test API connection first
        requests.get(f"{ATTENDANCE_API_URL}/health", timeout=2)
        return view_attendance_for_date(date.today().strftime('%Y-%m-%d'))
    except requests.exceptions.RequestException:
        flash("Attendance API is not running. Please start attendance_api.py on port 5001", "error")
        return render_template('index.html',
                           selected_date=date.today().strftime('%Y-%m-%d'),
                           available_dates=[],
                           attendance_data=[],
                           no_data=True,
                           today_date=date.today().strftime('%Y-%m-%d'))

def view_attendance_for_date(selected_date):
    """Shared logic for date-based attendance viewing"""
    try:
        datetime.strptime(selected_date, '%Y-%m-%d')  # Validate format
    except ValueError:
        selected_date = date.today().strftime('%Y-%m-%d')
        flash('Invalid date format', 'error')

    attendance_data = []
    try:
        response = requests.get(
            f"{ATTENDANCE_API_URL}/attendance/report?date={selected_date}",
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            attendance_data = response.json().get('records', [])
        elif response.status_code == 404:
            flash('No data for selected date', 'info')
    except requests.exceptions.RequestException as e:
        flash('Service temporarily unavailable', 'error')
        logging.error(f"API Error: {str(e)}")

    return render_template(
        'index.html',
        selected_date=selected_date,
        available_dates=get_available_dates(),
        attendance_data=attendance_data,
        no_data=not bool(attendance_data),
        today_date=date.today().strftime('%Y-%m-%d')  # Pass as string instead of date object
    )

@app.route('/view_attendance', methods=['POST'])
def view_attendance():
    return view_attendance_for_date(request.form.get('selected_date'))

@app.route('/export_attendance')
def export_attendance():
    selected_date = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    try:
        response = requests.get(
            f"{ATTENDANCE_API_URL}/attendance/export?date={selected_date}",
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return Response(
                response.content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=attendance_{selected_date}.csv'}
            )
    except requests.exceptions.RequestException as e:
        flash('Export service unavailable', 'error')
        logging.error(f"Export Error: {str(e)}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)