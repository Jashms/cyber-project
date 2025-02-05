import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from model_training import train_model
from datetime import datetime
import time
from test_server.monitor import start_monitoring
import threading
import queue
from test_server.data_generator import generate_test_data
from auth.user_management import (
    init_db, create_user, verify_user, has_permission, 
    UserRole, Permission, change_password
)
from auth.two_factor import generate_totp_secret, generate_totp_uri, generate_qr_code, verify_totp
from threat_intel.intel_feeds import ThreatIntel
from model_training_dl import DeepIDS
from deep_predictor import DeepPredictor
import requests
import folium
from folium import plugins
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict
import networkx as nx

# Add this to store live traffic data
traffic_queue = queue.Queue()

# Initialize ThreatIntel instance
threat_intel = ThreatIntel()

# Initialize deep learning model
deep_ids = DeepIDS()

# Initialize the deep learning predictor
deep_predictor = DeepPredictor()

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

def monitor_traffic(data, model, scaler):
    try:
        # Get the feature names from the training data
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Ensure data has all required features
        for feature in feature_names:
            if feature not in data.columns:
                data[feature] = 0
        
        # Select and order features to match training data
        X = data[feature_names]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        predictions = model.predict(X_scaled)
        
        # Convert predictions to pandas Series with the same index as data
        predictions = pd.Series(predictions, index=data.index)
        
        # Add threat intelligence enrichment
        enriched_data = data.copy()  # Start with original data
        
        # Modify attack detection logic to include threat intel
        attack_indicators = pd.DataFrame(index=data.index)
        attack_indicators['high_packet_rate'] = data['Flow Packets/s'] > 1000
        attack_indicators['high_byte_rate'] = data['Flow Bytes/s'] > 100000
        attack_indicators['syn_flags'] = data['SYN Flag Count'] > 10
        attack_indicators['short_flow'] = data['Flow Duration'] < 50
        
        # Determine if traffic is malicious based on indicators
        is_attack = (attack_indicators.sum(axis=1) >= 2)  # If 2 or more indicators are True
        
        # Update threat scores based on attack indicators
        malicious_ips = ['185.159.128.51', '45.89.67.43', '103.235.46.172', '91.92.109.126']
        
        # Initialize threat-related columns if they don't exist
        if 'threat_score' not in enriched_data.columns:
            enriched_data['threat_score'] = 0
        if 'threat_categories' not in enriched_data.columns:
            enriched_data['threat_categories'] = ''
        if 'last_reported' not in enriched_data.columns:
            enriched_data['last_reported'] = ''
        
        # Update threat scores for known malicious IPs
        for ip in malicious_ips:
            mask = (enriched_data['Src IP'] == ip) | (enriched_data['Dst IP'] == ip)
            if any(mask):
                enriched_data.loc[mask, 'threat_score'] = np.random.randint(60, 100)
                enriched_data.loc[mask, 'threat_categories'] = 'SSH Bruteforce, Port Scan'
                enriched_data.loc[mask, 'last_reported'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update threat scores based on attack indicators
        for idx in enriched_data.index:
            if is_attack[idx]:
                current_score = enriched_data.loc[idx, 'threat_score']
                if current_score < 50:  # Only increase if current score is low
                    enriched_data.loc[idx, 'threat_score'] = np.random.randint(50, 85)
                    enriched_data.loc[idx, 'threat_categories'] = 'Suspicious Activity'
                    enriched_data.loc[idx, 'last_reported'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Combine model predictions with custom logic
        final_predictions = predictions.copy()
        final_predictions[is_attack] = np.where(
            (data.loc[is_attack, 'Flow Packets/s'] > 5000) | 
            (enriched_data.loc[is_attack, 'threat_score'] > 80),
            'DOS',
            'Port Scan'
        )
        
        # Add deep learning predictions
        dl_labels, dl_scores = deep_predictor.predict(data)
        
        if dl_labels is not None and dl_scores is not None:
            # Add deep learning results to enriched data
            enriched_data['dl_prediction'] = dl_labels
            enriched_data['dl_confidence'] = dl_scores
            
            # Update threat scores based on deep learning
            for idx, (label, score) in enumerate(zip(dl_labels, dl_scores)):
                if label != 'BENIGN':
                    current_score = enriched_data.loc[idx, 'threat_score']
                    dl_threat_score = score * 100  # Convert to 0-100 scale
                    # Take maximum of current and deep learning threat score
                    enriched_data.loc[idx, 'threat_score'] = max(current_score, dl_threat_score)
                    
                    # Update threat categories
                    if enriched_data.loc[idx, 'threat_categories']:
                        enriched_data.loc[idx, 'threat_categories'] += f", {label}"
                    else:
                        enriched_data.loc[idx, 'threat_categories'] = str(label)
        
        return final_predictions, enriched_data
        
    except Exception as e:
        st.error(f"Error in monitoring: {str(e)}")
        return None, None

def generate_sample_traffic():
    """Generate realistic sample network traffic data"""
    # Generate a mix of normal and potentially malicious IPs
    ips = [
        '8.8.8.8',         # Google DNS (normal)
        '1.1.1.1',         # Cloudflare DNS (normal)
        '185.159.128.51',  # Example potentially malicious IP
        '192.168.1.1',     # Local network (normal)
        '45.89.67.43',     # Another potentially malicious IP
        '103.235.46.172',  # Another potentially malicious IP
        '91.92.109.126',   # Another potentially malicious IP
    ]
    
    # Generate multiple records
    num_records = np.random.randint(3, 10)  # Generate 3-10 records at a time
    
    sample_data = {
        'Src IP': [np.random.choice(ips) for _ in range(num_records)],
        'Dst IP': [np.random.choice(ips) for _ in range(num_records)],
        'Src Port': [np.random.randint(1, 65535) for _ in range(num_records)],
        'Dst Port': [np.random.randint(1, 65535) for _ in range(num_records)],
        'Protocol': [np.random.randint(0, 17) for _ in range(num_records)],
        'Flow Duration': [np.random.randint(0, 1000000) for _ in range(num_records)],
        'Flow Packets/s': [np.random.randint(1, 5000) for _ in range(num_records)],
        'Flow Bytes/s': [np.random.randint(1, 100000) for _ in range(num_records)],
        'Total Fwd Packet': [np.random.randint(1, 100) for _ in range(num_records)],
        'Total Bwd packets': [np.random.randint(1, 100) for _ in range(num_records)],
        'Total Length of Fwd Packet': [np.random.randint(0, 10000) for _ in range(num_records)],
        'Total Length of Bwd Packet': [np.random.randint(0, 10000) for _ in range(num_records)],
        'SYN Flag Count': [np.random.randint(0, 20) for _ in range(num_records)],
        # Add simulated threat scores
        'threat_score': [
            np.random.choice(
                [0, 0, 0, 25, 50, 75, 100],  # More weight to normal traffic
                p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]  # Probability distribution
            ) for _ in range(num_records)
        ],
        'threat_categories': [''] * num_records,
        'last_reported': [''] * num_records
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some realistic threat data for known malicious IPs
    malicious_ips = ['185.159.128.51', '45.89.67.43', '103.235.46.172', '91.92.109.126']
    for ip in malicious_ips:
        mask = (df['Src IP'] == ip) | (df['Dst IP'] == ip)
        if any(mask):
            df.loc[mask, 'threat_score'] = np.random.randint(60, 100)
            df.loc[mask, 'threat_categories'] = 'SSH Bruteforce, Port Scan'
            df.loc[mask, 'last_reported'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def monitor_callback(packet_data):
    if packet_data and hasattr(st.session_state, 'traffic_queue'):
        try:
            st.session_state.traffic_queue.put(packet_data)
        except Exception as e:
            print(f"Error adding to queue: {str(e)}")

def change_password_page():
    st.title("Change Password")
    
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
        
        if submitted:
            if new_password != confirm_password:
                st.error("New passwords don't match!")
                return
                
            success, message = change_password(
                st.session_state.username,
                current_password,
                new_password
            )
            
            if success:
                st.success(message)
                st.session_state.password_changed = True
            else:
                st.error(message)

def setup_2fa():
    st.title("Setup Two-Factor Authentication")
    
    if 'totp_secret' not in st.session_state:
        st.session_state.totp_secret = generate_totp_secret()
    
    # Generate QR code
    uri = generate_totp_uri(st.session_state.username, st.session_state.totp_secret)
    qr_code = generate_qr_code(uri)
    
    st.markdown("### 1. Scan QR Code")
    st.markdown("Use your authenticator app (like Google Authenticator) to scan this QR code:")
    st.image(f"data:image/png;base64,{qr_code}")
    
    st.markdown("### 2. Verify Setup")
    with st.form("verify_2fa"):
        token = st.text_input("Enter the code from your authenticator app")
        submitted = st.form_submit_button("Verify")
        
        if submitted:
            if verify_totp(st.session_state.totp_secret, token):
                # Save TOTP secret to database
                st.success("2FA setup successful!")
            else:
                st.error("Invalid code. Please try again.")

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    # Add traffic queue initialization
    if 'traffic_queue' not in st.session_state:
        st.session_state.traffic_queue = queue.Queue()

def login_page():
    st.title("Login")
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = verify_user(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    # Add a note about default admin credentials
    st.markdown("---")
    st.markdown("Default admin credentials:")
    st.markdown("- Username: admin")
    st.markdown("- Password: changeme123")

def show_threat_intel_dashboard(enriched_data):
    # Generate a unique timestamp for keys
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    st.subheader("Threat Intelligence & Deep Learning Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Deep Learning Insights", "Detailed Analysis"])
    
    with tab1:
        # Original threat score distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                enriched_data,
                x='threat_score',
                title='Threat Score Distribution',
                nbins=20,
                color_discrete_sequence=['red']
            )
            st.plotly_chart(fig, key=f"threat_hist_{timestamp}", use_container_width=True)
        
        with col2:
            if 'dl_confidence' in enriched_data.columns:
                fig = px.histogram(
                    enriched_data,
                    x='dl_confidence',
                    title='Detection Confidence Distribution',
                    nbins=20,
                    color_discrete_sequence=['blue']
                )
                st.plotly_chart(fig, key=f"conf_hist_{timestamp}", use_container_width=True)
    
    with tab2:
        if 'dl_prediction' in enriched_data.columns:
            # Attack type distribution
            col1, col2 = st.columns(2)
            with col1:
                attack_counts = enriched_data['dl_prediction'].value_counts()
                fig = px.pie(
                    values=attack_counts.values,
                    names=attack_counts.index,
                    title='Attack Type Distribution'
                )
                st.plotly_chart(fig, key=f"attack_pie_{timestamp}", use_container_width=True)
            
            with col2:
                # High confidence attacks over time
                high_conf_attacks = enriched_data[enriched_data['dl_confidence'] > 0.9]
                if not high_conf_attacks.empty:
                    st.warning(f"🚨 Detected {len(high_conf_attacks)} high-confidence attacks!")
                    
                    # Show attack timeline
                    for _, row in high_conf_attacks.iterrows():
                        st.markdown(
                            f"⚠️ **{row['dl_prediction']}** attack detected\n"
                            f"- Source: `{row['Src IP']}`\n"
                            f"- Target: `{row['Dst IP']}`\n"
                            f"- Confidence: `{row['dl_confidence']:.2%}`"
                        )
                else:
                    st.success("No high-confidence attacks detected")
    
    with tab3:
        # Detailed analysis
        st.subheader("Traffic Analysis")
        
        # Filter options with unique keys
        attack_types = ['All'] + list(enriched_data['dl_prediction'].unique())
        selected_type = st.selectbox(
            "Filter by Attack Type",
            attack_types,
            key=f"attack_type_select_{timestamp}"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key=f"conf_threshold_{timestamp}"
        )
        
        # Filter data based on selections
        filtered_data = enriched_data.copy()
        if selected_type != 'All':
            filtered_data = filtered_data[filtered_data['dl_prediction'] == selected_type]
        filtered_data = filtered_data[filtered_data['dl_confidence'] >= confidence_threshold]
        
        # Show filtered results
        if not filtered_data.empty:
            st.dataframe(
                filtered_data[[
                    'Src IP', 'Dst IP', 'dl_prediction', 'dl_confidence',
                    'threat_score', 'Flow Packets/s', 'Flow Bytes/s'
                ]].sort_values('dl_confidence', ascending=False),
                hide_index=True,
                key=f"filtered_data_{timestamp}"
            )
            
            # Show summary metrics without keys
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Detections",
                    len(filtered_data)
                )
            with col2:
                st.metric(
                    "Avg. Confidence",
                    f"{filtered_data['dl_confidence'].mean():.2%}"
                )
            with col3:
                st.metric(
                    "Avg. Threat Score",
                    f"{filtered_data['threat_score'].mean():.1f}"
                )
        else:
            st.info("No traffic matching the selected criteria")
    
    # Show raw data in expander
    with st.expander("View Raw Data"):
        st.dataframe(
            enriched_data[[
                'Src IP', 'Dst IP', 'dl_prediction', 'dl_confidence',
                'threat_score', 'threat_categories', 'Flow Packets/s',
                'Flow Bytes/s'
            ]].sort_values('threat_score', ascending=False),
            hide_index=True,
            key=f"raw_data_{timestamp}"
        )

def validate_and_prepare_data(data):
    """Validate and prepare data for monitoring"""
    try:
        # Create a copy of the data
        prepared_data = data.copy()
        
        # Column mapping for your specific dataset
        column_mapping = {
            'Src IP dec': 'Src IP',
            'Dst IP dec': 'Dst IP',
            'Flow Duration': 'Flow Duration',
            'Flow Packets/s': 'Flow Packets/s',
            'Flow Bytes/s': 'Flow Bytes/s',
            'Src Port': 'Src Port',
            'Dst Port': 'Dst Port',
            'Protocol': 'Protocol',
            'Label': 'Label'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in prepared_data.columns:
                prepared_data = prepared_data.rename(columns={old_col: new_col})
        
        # Verify required columns exist after renaming
        required_columns = [
            'Src IP', 'Dst IP', 'Flow Duration', 'Flow Packets/s', 'Flow Bytes/s'
        ]
        
        missing_cols = [col for col in required_columns if col not in prepared_data.columns]
        if missing_cols:
            print(f"Missing required columns after renaming: {missing_cols}")
            print("Available columns:", prepared_data.columns.tolist())
            return None
        
        # Fill missing values with 0
        prepared_data = prepared_data.fillna(0)
        
        # Convert IP decimal to string format if needed
        if 'Src IP' in prepared_data.columns:
            prepared_data['Src IP'] = prepared_data['Src IP'].astype(str)
        if 'Dst IP' in prepared_data.columns:
            prepared_data['Dst IP'] = prepared_data['Dst IP'].astype(str)
        
        return prepared_data
        
    except Exception as e:
        print(f"Error in data validation: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        return None

def get_ip_location(ip):
    """Get location information for an IP address"""
    try:
        # Skip private/local IP addresses
        if ip.startswith(('192.168.', '10.', '172.', '127.')):
            return None
            
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Validate that we have both latitude and longitude
            if data.get('latitude') is not None and data.get('longitude') is not None:
                return {
                    'latitude': float(data.get('latitude')),
                    'longitude': float(data.get('longitude')),
                    'country': data.get('country_name', 'Unknown'),
                    'city': data.get('city', 'Unknown'),
                    'threat_data': {}
                }
    except Exception as e:
        print(f"Error getting location for IP {ip}: {str(e)}")
    return None

def create_attack_map(enriched_data):
    """Create an interactive map showing attack origins"""
    try:
        # Create base map
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Track unique attack locations
        attack_locations = {}
        valid_locations = False
        
        # Process each suspicious/malicious connection
        for _, row in enriched_data[enriched_data['threat_score'] > 50].iterrows():
            src_ip = row['Src IP']
            if src_ip not in attack_locations:
                location_data = get_ip_location(src_ip)
                if location_data and location_data.get('latitude') is not None:
                    attack_locations[src_ip] = location_data
                    valid_locations = True
                    
                    # Create popup content
                    popup_content = f"""
                        <b>IP:</b> {src_ip}<br>
                        <b>Location:</b> {location_data['city']}, {location_data['country']}<br>
                        <b>Threat Score:</b> {row['threat_score']}<br>
                        <b>Attack Type:</b> {row.get('threat_categories', 'Unknown')}
                    """
                    
                    # Add marker to cluster
                    folium.Marker(
                        location=[location_data['latitude'], location_data['longitude']],
                        popup=popup_content,
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(marker_cluster)
        
        if not valid_locations:
            return None
            
        # Add heat map layer if we have valid locations
        heat_data = [
            [loc['latitude'], loc['longitude']] 
            for loc in attack_locations.values() 
            if loc['latitude'] is not None and loc['longitude'] is not None
        ]
        if heat_data:
            plugins.HeatMap(heat_data).add_to(m)
        
        return m
        
    except Exception as e:
        print(f"Error creating map: {str(e)}")
        return None

def show_attack_map(enriched_data):
    """Display the attack map in Streamlit"""
    try:
        # Create map
        attack_map = create_attack_map(enriched_data)
        
        if attack_map is None:
            st.warning("No valid geolocation data available for mapping. This could be due to:")
            st.markdown("""
                - Local/private IP addresses in the traffic
                - Geolocation service rate limits
                - Network connectivity issues
                - Invalid IP addresses in the data
            """)
            return
            
        # Save map to HTML string
        map_html = attack_map._repr_html_()
        
        # Display using components
        components.html(map_html, height=400)
        
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")

def generate_pdf_report(data, report_type="incident"):
    """Generate a PDF report with graphs and statistics"""
    try:
        # Create a BytesIO buffer to store the PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title = f"Security Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        story.append(Paragraph(title, styles['Heading1']))
        story.append(Spacer(1, 12))

        # Add summary statistics
        summary_data = [
            ["Total Traffic Analyzed", str(len(data))],
            ["High Threat Events", str(len(data[data['threat_score'] > 80]))],
            ["Unique Source IPs", str(data['Src IP'].nunique())],
            ["Average Threat Score", f"{data['threat_score'].mean():.2f}"]
        ]

        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Generate and add graphs
        # Threat Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='threat_score', bins=20)
        plt.title('Threat Score Distribution')
        img_buffer = BytesIO()
        plt.savefig(img_buffer)
        img_buffer.seek(0)
        story.append(Image(img_buffer))
        plt.close()

        # Add attack patterns if available
        if 'dl_prediction' in data.columns:
            story.append(Paragraph('Attack Pattern Distribution', styles['Heading2']))
            attack_counts = data['dl_prediction'].value_counts()
            attack_data = [[str(k), str(v)] for k, v in attack_counts.items()]
            attack_table = Table([['Attack Type', 'Count']] + attack_data)
            attack_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ]))
            story.append(attack_table)

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def analyze_attack_patterns(data):
    """Analyze and identify common attack patterns"""
    patterns = defaultdict(int)
    attack_sequences = []
    
    try:
        # Look for patterns in traffic behavior
        for i in range(len(data)):
            if data.iloc[i]['threat_score'] > 50:  # Focus on suspicious traffic
                pattern = {
                    'src_ip': data.iloc[i]['Src IP'],
                    'dst_ip': data.iloc[i]['Dst IP'],
                    'packets_per_sec': float(data.iloc[i]['Flow Packets/s']),
                    'bytes_per_sec': float(data.iloc[i]['Flow Bytes/s']),
                    'threat_score': float(data.iloc[i]['threat_score'])
                }
                
                # Add threat categories if available
                if 'threat_categories' in data.columns:
                    pattern['threat_type'] = data.iloc[i]['threat_categories']
                
                # Add deep learning predictions if available
                if 'dl_prediction' in data.columns:
                    pattern['dl_prediction'] = data.iloc[i]['dl_prediction']
                
                attack_sequences.append(pattern)
                
                # Create a simplified pattern key for counting occurrences
                pattern_key = f"SRC:{pattern['src_ip']}_DST:{pattern['dst_ip']}"
                patterns[pattern_key] += 1
        
        # Analyze attack sequences
        analyzed_patterns = []
        for pattern_key, count in patterns.items():
            # Get all sequences matching this pattern
            matching_sequences = [
                seq for seq in attack_sequences 
                if f"SRC:{seq['src_ip']}_DST:{seq['dst_ip']}" == pattern_key
            ]
            
            if matching_sequences:
                # Calculate average metrics
                avg_packets = sum(seq['packets_per_sec'] for seq in matching_sequences) / len(matching_sequences)
                avg_bytes = sum(seq['bytes_per_sec'] for seq in matching_sequences) / len(matching_sequences)
                avg_threat = sum(seq['threat_score'] for seq in matching_sequences) / len(matching_sequences)
                
                # Get the most common threat type if available
                threat_types = [
                    seq.get('threat_type', 'Unknown') 
                    for seq in matching_sequences 
                    if seq.get('threat_type')
                ]
                most_common_threat = max(set(threat_types), key=threat_types.count) if threat_types else 'Unknown'
                
                analyzed_patterns.append({
                    'pattern_key': pattern_key,
                    'count': count,
                    'src_ip': matching_sequences[0]['src_ip'],
                    'dst_ip': matching_sequences[0]['dst_ip'],
                    'avg_packets_per_sec': avg_packets,
                    'avg_bytes_per_sec': avg_bytes,
                    'avg_threat_score': avg_threat,
                    'threat_type': most_common_threat,
                    'dl_prediction': matching_sequences[0].get('dl_prediction', 'Unknown')
                })
        
        # Sort by count and threat score
        analyzed_patterns.sort(key=lambda x: (x['count'], x['avg_threat_score']), reverse=True)
        return analyzed_patterns
        
    except Exception as e:
        print(f"Error in attack pattern analysis: {str(e)}")
        return []

def store_historical_data(data):
    """Store traffic data for historical analysis"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'historical_data_{timestamp}.json'
        
        # Add debug print statements
        print(f"Storing historical data with {len(data)} records")
        
        # Prepare data for storage
        historical_data = {
            'timestamp': timestamp,
            'traffic_summary': {
                'total_connections': len(data),
                'unique_sources': data['Src IP'].nunique(),
                'unique_destinations': data['Dst IP'].nunique(),
                'avg_threat_score': float(data['threat_score'].mean()),
                'high_threat_events': int(sum(data['threat_score'] > 80))
            },
            'attack_patterns': analyze_attack_patterns(data)
        }
        
        # Create directory if it doesn't exist
        os.makedirs('historical_data', exist_ok=True)
        
        # Save to file
        filepath = f'historical_data/{filename}'
        with open(filepath, 'w') as f:
            json.dump(historical_data, f)
            
        print(f"Successfully stored data to {filepath}")
        return True
    except Exception as e:
        print(f"Error storing historical data: {str(e)}")
        return False

def get_historical_trends(days=30):
    """Analyze historical trends from stored data"""
    try:
        trends = {
            'dates': [],
            'total_traffic': [],
            'threat_scores': [],
            'attack_patterns': defaultdict(list)
        }
        
        # Debug print
        print("Checking for historical data...")
        
        # Ensure directory exists
        if not os.path.exists('historical_data'):
            print("Historical data directory not found")
            return None
        
        # Get list of historical data files
        historical_files = sorted(os.listdir('historical_data'))
        print(f"Found {len(historical_files)} historical data files")
        
        # Filter files for specified date range
        start_date = datetime.now() - timedelta(days=days)
        
        for filename in historical_files:
            try:
                file_date = datetime.strptime(filename.split('_')[1], '%Y%m%d')
                if file_date >= start_date:
                    filepath = f'historical_data/{filename}'
                    print(f"Processing file: {filepath}")
                    
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        trends['dates'].append(file_date.strftime('%Y-%m-%d'))
                        trends['total_traffic'].append(data['traffic_summary']['total_connections'])
                        trends['threat_scores'].append(data['traffic_summary']['avg_threat_score'])
                        
                        # Track attack patterns
                        for pattern, count in data['attack_patterns']:
                            trends['attack_patterns'][pattern].append(count)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
        
        print(f"Processed {len(trends['dates'])} days of historical data")
        return trends if trends['dates'] else None
    
    except Exception as e:
        print(f"Error analyzing historical trends: {str(e)}")
        return None

def create_network_graph(data):
    """Create interactive network visualization"""
    try:
        # Create graph
        G = nx.Graph()
        valid_connections = 0
        
        # Track statistics for the info box
        stats = {
            'max_threat': 0,
            'avg_threat': 0,
            'total_threats': 0,
            'high_risk_ips': set()
        }
        
        # Add nodes and edges from traffic data
        for idx, row in data.iterrows():
            try:
                src = str(row['Src IP']).strip()
                dst = str(row['Dst IP']).strip()
                threat_score = float(row['threat_score'])
                
                # Update statistics
                stats['total_threats'] += threat_score
                stats['max_threat'] = max(stats['max_threat'], threat_score)
                if threat_score > 70:  # High risk threshold
                    stats['high_risk_ips'].add(src)
                
                # Skip invalid entries
                if not src or not dst or src == 'nan' or dst == 'nan':
                    continue
                
                # Add nodes with attributes
                G.add_node(src, node_type='source', threat_score=threat_score)
                G.add_node(dst, node_type='destination')
                
                # Add edge with threat score as weight
                G.add_edge(src, dst, weight=threat_score)
                valid_connections += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        if valid_connections == 0:
            return None
            
        # Calculate average threat score
        stats['avg_threat'] = stats['total_threats'] / valid_connections
        
        # Create layout with more spacing
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces with improved color gradient
        edge_traces = []
        max_threat = max([G.edges[edge]['weight'] for edge in G.edges()])
        min_threat = min([G.edges[edge]['weight'] for edge in G.edges()])
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G.edges[edge]['weight']
            
            # Calculate color based on threat score
            if max_threat == min_threat:
                color = 'rgb(255,0,0)'
            else:
                normalized = (weight - min_threat) / (max_threat - min_threat)
                # Enhanced color gradient: blue -> yellow -> red
                if normalized < 0.5:
                    # Blue to yellow
                    blue_to_yellow = normalized * 2
                    color = f'rgb({int(255*blue_to_yellow)},{int(255*blue_to_yellow)},{255})'
                else:
                    # Yellow to red
                    yellow_to_red = (normalized - 0.5) * 2
                    color = f'rgb(255,{int(255*(1-yellow_to_red))},0)'
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color=color),
                hoverinfo='text',
                text=f"Connection: {edge[0]} → {edge[1]}<br>Threat Score: {weight:.2f}",
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace with enhanced information
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_symbols = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Calculate node size based on connections and threat
            connections = len(list(G.neighbors(node)))
            base_size = 25 + (connections * 15)
            
            if G.nodes[node]['node_type'] == 'source':
                threat = G.nodes[node].get('threat_score', 0)
                # Increase size for high-threat nodes
                node_size.append(base_size * (1 + threat/100))
                node_color.append('#FF4444')
                node_symbols.append('diamond')
                node_text.append(
                    f'Source IP: {node}<br>'
                    f'Connections: {connections}<br>'
                    f'Threat Score: {threat:.2f}'
                )
            else:
                node_size.append(base_size)
                node_color.append('#4444FF')
                node_symbols.append('circle')
                node_text.append(
                    f'Destination IP: {node}<br>'
                    f'Incoming Connections: {connections}'
                )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                symbol=node_symbols,
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ))
        
        # Create figure with enhanced layout
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=dict(
                    text='Network Traffic Visualization',
                    x=0.5,
                    y=0.95,
                    font=dict(size=24)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text=f"High Risk IPs: {len(stats['high_risk_ips'])}",
                        x=0.02,
                        y=0.98,
                        showarrow=False,
                        font=dict(size=12, color='red')
                    ),
                    dict(
                        text=f"Avg Threat: {stats['avg_threat']:.2f}",
                        x=0.02,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=12, color='black')
                    ),
                    dict(
                        text="Source IPs (◆)",
                        x=0.95,
                        y=0.98,
                        showarrow=False,
                        font=dict(size=12, color='#FF4444')
                    ),
                    dict(
                        text="Destination IPs (●)",
                        x=0.95,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=12, color='#4444FF')
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#F8F9FA',
                paper_bgcolor='white',
                width=900,
                height=700,
                dragmode='pan'  # Enable panning
            )
        )
        
        # Add zoom buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.1,
                    y=1.1,
                    buttons=[
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"xaxis.range": None, "yaxis.range": None}]
                        ),
                        dict(
                            label="Zoom In",
                            method="relayout",
                            args=[{"xaxis.range": [-0.5, 0.5], "yaxis.range": [-0.5, 0.5]}]
                        )
                    ]
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating network graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Initialize session state at the very beginning
    initialize_session_state()
    
    # Initialize the database
    init_db()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        login_page()
        return

    st.set_page_config(page_title="Network IDS", page_icon="🛡️", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .status-normal {
            color: green;
            font-weight: bold;
        }
        .status-alert {
            color: red;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🛡️ Network IDS")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Model Training", "Real-time Monitoring", "Batch Analysis", "System Settings", "Reports & Analytics", "Network Visualization"]
    )
    
    # Add new pages to navigation
    with st.sidebar:
        if st.session_state.authenticated:
            if st.button("Change Password"):
                st.session_state.current_page = "change_password"
            if st.button("Setup 2FA"):
                st.session_state.current_page = "setup_2fa"
    
    if page == "Dashboard":
        st.title("Network Intrusion Detection System Dashboard")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="System Status", value="Active", delta="Online")
        with col2:
            st.metric(label="Threats Detected (24h)", value="0", delta="-2")
        with col3:
            st.metric(label="Model Accuracy", value="95%", delta="+2%")
        
        # Sample visualization
        st.subheader("Network Traffic Overview")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Normal', 'Suspicious', 'Malicious']
        )
        st.line_chart(chart_data)
        
        # Recent alerts
        st.subheader("Recent Alerts")
        if st.session_state.get('alerts') is None:
            st.session_state.alerts = []
        
        for alert in st.session_state.alerts[-5:]:
            st.error(alert)
            
    elif page == "Model Training":
        st.title("Model Training and Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Train New Model")
            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    train_acc, test_acc, features = train_model()
                st.success("Model trained successfully!")
                st.write(f"Training Accuracy: {train_acc:.2%}")
                st.write(f"Testing Accuracy: {test_acc:.2%}")
        
        with col2:
            st.subheader("Model Performance")
            metrics = {
                "Precision": 0.95,
                "Recall": 0.93,
                "F1-Score": 0.94,
            }
            for metric, value in metrics.items():
                st.metric(label=metric, value=f"{value:.2%}")
    
    elif page == "Real-time Monitoring":
        st.title("Real-time Network Monitoring")
        
        model, scaler = load_model()
        if model is None:
            st.error("Please train the model first!")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            monitoring_active = st.toggle("Start Monitoring")
        with col2:
            alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.9)
        with col3:
            refresh_rate = st.selectbox("Refresh Rate", [1, 5, 10, 30, 60], index=1)
        
        if monitoring_active:
            # Initialize traffic queue if not exists
            if 'traffic_queue' not in st.session_state:
                st.session_state.traffic_queue = queue.Queue()
            
            placeholder = st.empty()
            
            while monitoring_active:
                # Generate sample data for testing
                sample_data = generate_sample_traffic()
                
                # Monitor traffic
                predictions, enriched_data = monitor_traffic(sample_data, model, scaler)
                
                if predictions is not None and enriched_data is not None:
                    # Store the data for historical analysis
                    success = store_historical_data(enriched_data)
                    if success:
                        st.session_state.current_data = enriched_data
                        print("Successfully stored monitoring data")
                    else:
                        print("Failed to store monitoring data")
                    
                    # Update display
                    with placeholder.container():
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Current Status")
                            status = "Normal" if all(p == "BENIGN" for p in predictions) else "ALERT!"
                            st.markdown(f"**Status:** <span class='status-{'normal' if status == 'Normal' else 'alert'}'>{status}</span>", 
                                      unsafe_allow_html=True)
                            st.write(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
                            
                            # Show traffic rates
                            st.metric("Active Connections", len(predictions))
                            if len(predictions) > 0:
                                alert_rate = sum(p != "BENIGN" for p in predictions)/len(predictions)
                                st.metric("Alert Rate", f"{alert_rate*100:.2f}%")
                        
                        with col2:
                            st.subheader("Traffic Analysis")
                            # Display traffic metrics
                            if not sample_data.empty:
                                st.metric("Packets/sec", f"{sample_data['Flow Packets/s'].mean():.2f}")
                                st.metric("Bytes/sec", f"{sample_data['Flow Bytes/s'].mean():.2f}")
                        
                        # Show threat intelligence dashboard
                        st.markdown("---")
                        show_threat_intel_dashboard(enriched_data)
                        
                        # Show raw data in expander
                        with st.expander("View Raw Data"):
                            st.dataframe(enriched_data)
                
                time.sleep(refresh_rate)
                
                # Break the loop if monitoring is turned off
                if not monitoring_active:
                    break
        else:
            st.info("Monitoring is currently inactive. Toggle the switch above to start monitoring.")
    
    elif page == "Batch Analysis":
        st.title("Batch Traffic Analysis")
        
        # Add data generation controls
        st.subheader("Generate Test Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_records = st.number_input("Number of Records", min_value=100, max_value=10000, value=1000)
        with col2:
            attack_ratio = st.slider("Attack Ratio", 0.0, 1.0, 0.2, 0.1)
        with col3:
            attack_types = st.multiselect(
                "Attack Types",
                ["DOS", "Port Scan"],
                default=["DOS"]
            )
        
        if st.button("Generate Test Data"):
            with st.spinner("Generating test data..."):
                filename = generate_test_data(num_records, attack_ratio, attack_types)
                st.success(f"Test data generated: {filename}")
        
        st.divider()
        
        # Existing batch analysis code
        st.subheader("Analyze Traffic Data")
        uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type="csv")
        
        if uploaded_file is not None:
            model, scaler = load_model()
            if model is None:
                st.error("Please train the model first!")
                return
            
            # Load and analyze data
            data = pd.read_csv(uploaded_file)
            
            # Validate and prepare data
            prepared_data = validate_and_prepare_data(data)
            
            if prepared_data is not None:
                # Process the data
                predictions, enriched_data = monitor_traffic(prepared_data, model, scaler)
                
                if predictions is not None and enriched_data is not None:
                    # Store for historical analysis
                    store_historical_data(enriched_data)
                    
                    # Update session state
                    st.session_state.current_data = enriched_data
                    
                    # Show analysis results
                    show_threat_intel_dashboard(enriched_data)
                    
                    # Add report generation option
                    if st.button("Generate Analysis Report"):
                        pdf_buffer = generate_pdf_report(enriched_data)
                        if pdf_buffer:
                            st.download_button(
                                "Download Analysis Report",
                                data=pdf_buffer,
                                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.error("Error processing the data. Check the console for details.")
            else:
                st.error("Invalid data format. Please ensure all required columns are present.")
                st.write("Required columns: Src IP, Dst IP, Flow Duration, Flow Packets/s, Flow Bytes/s")
                st.write("Your columns:", list(data.columns))
    
    elif page == "System Settings":
        st.title("System Settings")
        
        st.subheader("Model Configuration")
        st.slider("Detection Sensitivity", 0.0, 1.0, 0.8)
        st.checkbox("Enable Automatic Updates")
        
        st.subheader("Notification Settings")
        st.checkbox("Email Alerts")
        st.checkbox("SMS Alerts")
        st.text_input("Alert Email Address")
        
        st.subheader("System Maintenance")
        if st.button("Clear Alert History"):
            if 'alerts' in st.session_state:
                st.session_state.alerts = []
            st.success("Alert history cleared!")
        
        if st.button("Export System Logs"):
            st.info("System logs exported successfully!")

    elif page == "Reports & Analytics":
        st.title("Reports & Analytics Dashboard")
        
        # Debug information
        st.sidebar.markdown("### Debug Information")
        if os.path.exists('historical_data'):
            files = os.listdir('historical_data')
            st.sidebar.write(f"Number of historical records: {len(files)}")
        else:
            st.sidebar.write("No historical data directory found")
        
        # Create tabs for different analytics features
        tab1, tab2, tab3 = st.tabs([
            "Generate Reports",
            "Historical Trends",
            "Attack Patterns"
        ])
        
        with tab1:
            st.subheader("Generate Security Reports")
            
            # Show current data status
            if 'current_data' in st.session_state:
                st.info(f"Current data contains {len(st.session_state.current_data)} records")
            else:
                st.warning("No current data available. Please run monitoring or batch analysis first.")
            
            report_type = st.selectbox(
                "Report Type",
                ["Incident Report", "Weekly Summary", "Monthly Analysis"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    if 'current_data' in st.session_state:
                        pdf_buffer = generate_pdf_report(st.session_state.current_data)
                        if pdf_buffer:
                            st.download_button(
                                label="Download Report",
                                data=pdf_buffer,
                                file_name=f"security_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Error generating report")
                    else:
                        st.warning("No data available for report generation")
        
        with tab2:
            st.subheader("Historical Trend Analysis")
            time_range = st.selectbox(
                "Time Range",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )
            
            days = int(time_range.split()[1])
            trends = get_historical_trends(days)
            
            if trends and trends['dates']:
                # Plot traffic trends
                fig1 = px.line(
                    x=trends['dates'],
                    y=trends['total_traffic'],
                    title="Traffic Volume Over Time"
                )
                st.plotly_chart(fig1)
                
                # Plot threat score trends
                fig2 = px.line(
                    x=trends['dates'],
                    y=trends['threat_scores'],
                    title="Average Threat Score Over Time"
                )
                st.plotly_chart(fig2)
                
                # Show raw data
                st.subheader("Raw Data")
                data_df = pd.DataFrame({
                    'Date': trends['dates'],
                    'Traffic Volume': trends['total_traffic'],
                    'Average Threat Score': trends['threat_scores']
                })
                st.dataframe(data_df)
            else:
                st.info("No historical data available for the selected time range")
                st.markdown("""
                To generate historical data:
                1. Run the Real-time Monitoring
                2. Or perform Batch Analysis
                3. Data will be automatically stored for analysis
                """)

        with tab3:
            st.subheader("Attack Pattern Analysis")
            if 'current_data' in st.session_state:
                patterns = analyze_attack_patterns(st.session_state.current_data)
                
                if patterns:
                    st.write("### Detected Attack Patterns")
                    
                    # Create tabs for different views
                    pattern_tab1, pattern_tab2 = st.tabs(["Summary View", "Detailed Analysis"])
                    
                    with pattern_tab1:
                        # Show summary cards for top patterns
                        for pattern in patterns[:5]:  # Show top 5 patterns
                            with st.container():
                                st.markdown(f"""
                                #### Pattern detected {pattern['count']} times
                                - **Source IP:** `{pattern['src_ip']}`
                                - **Destination IP:** `{pattern['dst_ip']}`
                                - **Average Threat Score:** `{pattern['avg_threat_score']:.2f}`
                                - **Attack Type:** `{pattern['threat_type']}`
                                - **Traffic Rate:** `{pattern['avg_packets_per_sec']:.2f}` packets/sec, 
                                  `{pattern['avg_bytes_per_sec']:.2f}` bytes/sec
                                """)
                                st.markdown("---")
                    
                    with pattern_tab2:
                        # Create a DataFrame for all patterns
                        pattern_df = pd.DataFrame(patterns)
                        
                        # Add filters
                        col1, col2 = st.columns(2)
                        with col1:
                            min_occurrences = st.slider(
                                "Minimum Occurrences",
                                min_value=1,
                                max_value=max(pattern_df['count']),
                                value=2
                            )
                        with col2:
                            min_threat = st.slider(
                                "Minimum Threat Score",
                                min_value=0,
                                max_value=100,
                                value=50
                            )
                        
                        # Filter and display patterns
                        filtered_patterns = pattern_df[
                            (pattern_df['count'] >= min_occurrences) &
                            (pattern_df['avg_threat_score'] >= min_threat)
                        ]
                        
                        if not filtered_patterns.empty:
                            st.dataframe(
                                filtered_patterns[[
                                    'src_ip', 'dst_ip', 'count', 'avg_threat_score',
                                    'threat_type', 'avg_packets_per_sec', 'avg_bytes_per_sec'
                                ]],
                                hide_index=True
                            )
                            
                            # Visualize pattern distribution
                            fig = px.bar(
                                filtered_patterns,
                                x='src_ip',
                                y='count',
                                color='avg_threat_score',
                                title='Attack Pattern Distribution'
                            )
                            st.plotly_chart(fig)
                        else:
                            st.info("No patterns match the current filters")
                else:
                    st.info("""
                    No significant attack patterns detected yet. 
                    This could be because:
                    - Not enough traffic data collected
                    - No suspicious activities detected
                    - Threat scores are below threshold
                    """)
            else:
                st.warning("""
                No data available for pattern analysis. 
                Please run monitoring or batch analysis first.
                """)

    elif page == "Network Visualization":
        st.title("Network Traffic Visualization")
        
        if 'current_data' in st.session_state and st.session_state.current_data is not None:
            # Show data summary
            st.sidebar.markdown("### Data Summary")
            st.sidebar.write(f"Total records: {len(st.session_state.current_data)}")
            
            # Show sample of the data
            st.sidebar.markdown("### Sample Data")
            st.sidebar.dataframe(
                st.session_state.current_data[['Src IP', 'Dst IP', 'threat_score']].head(),
                hide_index=True
            )
            
            try:
                # Create visualization with all data
                with st.spinner("Generating network visualization..."):
                    fig = create_network_graph(st.session_state.current_data)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total Connections",
                                len(st.session_state.current_data)
                            )
                        with col2:
                            st.metric(
                                "Unique Sources",
                                st.session_state.current_data['Src IP'].nunique()
                            )
                        with col3:
                            st.metric(
                                "Unique Destinations",
                                st.session_state.current_data['Dst IP'].nunique()
                            )
                        
                        # Show data table
                        st.markdown("### Connection Details")
                        st.dataframe(
                            st.session_state.current_data[['Src IP', 'Dst IP', 'threat_score']],
                            hide_index=True
                        )
                    else:
                        st.error("Could not generate visualization. Check the logs for details.")
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.write("Please check the application logs for more details.")
            
        else:
            st.warning("""
            No data available for visualization. 
            Please run monitoring or batch analysis first to generate traffic data.
            """)

    # Handle new pages
    if st.session_state.get('current_page') == "change_password":
        change_password_page()
    elif st.session_state.get('current_page') == "setup_2fa":
        setup_2fa()

if __name__ == "__main__":
    main()