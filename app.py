import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from functools import wraps

import bcrypt
import mysql.connector
import paho.mqtt.client as mqtt
from flask import (Flask, flash, jsonify, redirect, render_template,
                  request, send_from_directory, session, url_for, make_response, abort)
import os
import json
from datetime import datetime, timedelta
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit

# Import the prediction service
from enhanced_prediction_service import enhanced_prediction_service

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # Required for session management

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

socketio = SocketIO(app, cors_allowed_origins="*")

# Database configuration
db_config = {
    'host': 'database.xetf.my.id',  # Replace with your database host
    'user': 'root',  # Replace with your database username
    'password': 'Ehetenandayo123',  # Replace with your database password
    'database': 'mqttwebcredential'  # Replace with your database name
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

MQTT_BROKER = 'eclipse.xetf.my.id'
MQTT_PORT = 1883
MQTT_TOPIC = 'data/sensordata'
MQTT_USERNAME = 'stressed_user'
MQTT_PASSWORD = 'connecting'

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get user from database
            cursor.execute('SELECT * FROM Users WHERE username = %s', (username,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                session['user'] = username
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error='Invalid username or password')
                
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return render_template('login.html', error='Database error occurred')
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    return render_template('login.html')

# API Login route for Android app

@app.route('/api/history', methods=['GET'])
@cross_origin()
def get_logs():
    try:
        # Get query parameters with defaults
        limit = int(request.args.get('limit', 100))  # Default to 100 entries
        offset = int(request.args.get('offset', 0))
        
        # Date range filtering
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Read the log file
        log_file = os.path.join('logs', 'mqtt_log.json')
        if not os.path.exists(log_file):
            return jsonify({'status': 'error', 'message': 'Log file not found'}), 404
            
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_logs = []
            start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
            end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
            
            for log in logs:
                try:
                    log_dt = datetime.fromisoformat(log['timestamp'])
                    if start_dt <= log_dt <= end_dt:
                        filtered_logs.append(log)
                except (KeyError, ValueError):
                    continue
            logs = filtered_logs

        # Sort logs by timestamp descending (newest first)
        try:
            logs.sort(key=lambda log: datetime.fromisoformat(log['timestamp']), reverse=True)
        except Exception as e:
            app.logger.error(f"Error sorting logs: {e}")

        # Apply pagination
        total_logs = len(logs)
        paginated_logs = logs[offset:offset + limit]
        
        return jsonify({
            'status': 'success',
            'data': paginated_logs,
            'pagination': {
                'total': total_logs,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_logs
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error fetching logs: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
@cross_origin()
def api_login():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        return response
        
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400
    
    username = data['username']
    password = data['password']
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user from database
        cursor.execute('SELECT * FROM Users WHERE username = %s', (username,))
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            # Return success response with user data (excluding password)
            user_data = {k: v for k, v in user.items() if k != 'password'}
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'user': user_data
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401
            
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return jsonify({'status': 'error', 'message': 'Database error occurred'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Serve index.html (now protected)
@app.route('/')
@login_required
def index():
    return send_from_directory('templates', 'index.html')

# Serve static files
@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)


latest_payload = {}
# Store last 100 readings (adjust as needed)
history_buffer = deque(maxlen=100)
data_lock = threading.Lock()  # Lock for thread-safe access to shared data

def on_message(client, userdata, msg):
    global latest_payload, latest_mqtt_data
    with data_lock:
        try:
            data = json.loads(msg.payload.decode())
            # No need for a separate timestamp variable here, as it's handled in the log entry
            latest_payload = data  # Store the complete payload for immediate use

            # Prepare data for logging, this will be picked up by the logging thread
            latest_mqtt_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }

            # Calculate prediction
            prediction = enhanced_prediction_service.predict_co2(data)

            # Emit both sensor data and prediction to the frontend
            socketio.emit('sensor_data', {'data': data, 'prediction': prediction})

        except json.JSONDecodeError:
            print(f"Could not decode MQTT message: {msg.payload.decode()}")
        except Exception as e:
            print(f"An error occurred in on_message: {e}")

mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")

# Start MQTT loop in background
def mqtt_loop():
    mqtt_client.loop_forever()

import threading
import os
from datetime import datetime
import json
import time

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Store the latest MQTT data
latest_mqtt_data = {}

def log_mqtt_data():
    """Logs the latest MQTT data to a file every 60 seconds, with corruption handling."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    log_file_path = os.path.join(log_dir, 'mqtt_log.json')
    os.makedirs(log_dir, exist_ok=True)

    if latest_mqtt_data:
        try:
            with data_lock:
                logs = []
                # Try to read existing data
                try:
                    with open(log_file_path, 'r') as f:
                        # Check if file is empty to avoid JSONDecodeError
                        if os.path.getsize(log_file_path) > 0:
                            logs = json.load(f)
                except FileNotFoundError:
                    # File doesn't exist, we'll create it silently
                    pass
                except json.JSONDecodeError:
                    # File is corrupt, back it up and start a new one
                    print(f"Log file {log_file_path} is corrupt. Backing up and starting new log.")
                    backup_path = f"{log_file_path}.{int(time.time())}.bak"
                    try:
                        os.rename(log_file_path, backup_path)
                    except OSError as e:
                        print(f"Could not back up corrupt log file: {e}")
                    logs = [] # Start with an empty list

                # Append new data and write back
                logs.append(latest_mqtt_data)

                with open(log_file_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                print(f"Logged MQTT data to {log_file_path}")

        except Exception as e:
            print(f"Error in logging thread: {e}")

def mqtt_logging_loop():
    """Background thread to log MQTT data every minute"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    log_file_path = os.path.join(log_dir, 'mqtt_log.json')
    os.makedirs(log_dir, exist_ok=True)

    while True:
        time.sleep(60) # Wait for the interval
        
        if not latest_mqtt_data:
            continue # Nothing to log

        try:
            with data_lock: # Ensure data isn't changed while we're logging
                logs = []
                # Step 1: Try to read existing data
                try:
                    with open(log_file_path, 'r') as f:
                        # Check if file is empty to avoid JSONDecodeError
                        if os.path.getsize(log_file_path) > 0:
                            logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = [] # Overwrite if not a list
                except FileNotFoundError:
                    # File doesn't exist, we'll create it silently
                    pass
                except json.JSONDecodeError:
                    # Step 2: Handle corrupt file
                    print(f"Log file {log_file_path} is corrupt. Backing up and starting new log.")
                    backup_path = f"{log_file_path}.{int(time.time())}.bak"
                    try:
                        os.rename(log_file_path, backup_path)
                    except OSError as e:
                        print(f"Could not back up corrupt log file: {e}")
                    logs = [] # Start with an empty list

                # Step 3: Append new data and write back
                logs.append(latest_mqtt_data)

                with open(log_file_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                print(f"Logged MQTT data to {log_file_path}")

        except Exception as e:
            print(f"Error in logging thread: {e}")

# Start MQTT client loop in background
threading.Thread(target=mqtt_loop, daemon=True).start()

# Start MQTT logging in background
threading.Thread(target=mqtt_logging_loop, daemon=True).start()

@app.route('/status')
@login_required
def status():
    # Check if MQTT client is connected
    is_connected = mqtt_client.is_connected() if hasattr(mqtt_client, 'is_connected') else False
    
    # Get prediction if we have data
    prediction = None
    if latest_mqtt_data and 'data' in latest_mqtt_data:
        try:
            prediction = enhanced_prediction_service.predict_co2(latest_mqtt_data['data'])
        except Exception as e:
            print(f"Error making prediction: {e}")
    
    return {
        'mqtt_connected': is_connected,
        'prediction': prediction
    }, 200

logging.basicConfig(level=logging.DEBUG)

@app.route('/update', methods=['POST'])
def update_payload():
    global latest_payload
    data = request.json  # Get JSON data from the Android client
    if data:
        latest_payload = data
        logging.debug(f"Received payload: {latest_payload}")
        return jsonify({"message": "Payload updated successfully"}), 200
    else:
        return jsonify({"error": "No data received"}), 400


@app.route('/history', methods=['GET'])
def get_history():
    if latest_payload:
        try:
            # Transform the MQTT data into the format Android expects
            history_data = []
            timestamp = datetime.now().isoformat()
            
            # Create an entry for each gas type in the payload
            for gas_type, value in latest_payload.items():
                if value:  # Only include non-empty values
                    history_data.append({
                        "timestamp": timestamp,
                        "gas_type": gas_type,
                        "current_value": str(value)
                    })
            
            return jsonify(history_data), 200
        except Exception as e:
            logging.error(f"Error formatting history data: {e}")
            return jsonify({"error": "Data formatting error"}), 500
    else:
        return jsonify({"error": "No data available"}), 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
