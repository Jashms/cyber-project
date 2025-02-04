from flask import Flask, jsonify, request
import logging
from datetime import datetime
import csv
import os

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a CSV file to log traffic
def log_request():
    timestamp = datetime.now()
    src_ip = request.remote_addr
    src_port = request.environ.get('REMOTE_PORT')
    dst_port = 5001  # Changed from 5000 to 5001
    
    # Create log entry
    log_entry = {
        'Src IP dec': int(src_ip.replace('.', '')),
        'Src Port': src_port,
        'Dst Port': dst_port,
        'Protocol': 6,  # TCP
        'Timestamp': timestamp.strftime('%H:%M:%S'),
        'Flow Duration': 0,
        'Total Fwd Packet': 1,
        'Total Bwd packets': 0,
        # Add other required fields with default values
    }
    
    # Write to CSV
    with open('live_traffic.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if os.stat('live_traffic.csv').st_size == 0:
            writer.writeheader()
        writer.writerow(log_entry)

@app.route('/')
def home():
    log_request()
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/api/data')
def get_data():
    log_request()
    return jsonify({"data": "sample data"})

@app.route('/api/login', methods=['POST'])
def login():
    log_request()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Create empty log file
    with open('live_traffic.csv', 'w', newline='') as f:
        pass
    
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed port to 5001 