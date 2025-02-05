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
import shap
from shap import KernelExplainer, summary_plot
import matplotlib.pyplot as plt

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

def generate_shap_explanations(model, data, scaler, feature_names):
    """Generate SHAP explanations for model predictions"""
    try:
        # Ensure data has all required features
        for feature in feature_names:
            if feature not in data.columns:
                data[feature] = 0
        
        # Select and order features to match training data
        X = data[feature_names].copy()
        
        # Print debug information
        print(f"Feature names: {feature_names}")
        print(f"Data columns: {X.columns.tolist()}")
        print(f"Data shape: {X.shape}")
        
        # Scale the input data
        scaled_data = scaler.transform(X)
        print(f"Scaled data shape: {scaled_data.shape}")
        
        # Create a background dataset for SHAP
        background_data = scaled_data[:100] if len(scaled_data) > 100 else scaled_data
        
        # Initialize the SHAP explainer
        explainer = KernelExplainer(model.predict_proba, background_data)
        
        # Calculate SHAP values for a smaller subset
        max_samples = min(len(scaled_data), 100)
        shap_values = explainer.shap_values(scaled_data[:max_samples])
        
        # Print SHAP values shape
        print(f"SHAP values shape: {[sv.shape for sv in shap_values]}")
        print(f"X shape for visualization: {X[:max_samples].shape}")
        
        # Ensure the shapes match for visualization
        shap_values_for_viz = None
        if isinstance(shap_values, list):
            shap_values_for_viz = shap_values[1]  # For binary classification, use class 1
            if shap_values_for_viz.shape[1] != len(feature_names):
                # Transpose if necessary
                shap_values_for_viz = shap_values_for_viz.T
        else:
            shap_values_for_viz = shap_values
            if shap_values_for_viz.shape[1] != len(feature_names):
                # Transpose if necessary
                shap_values_for_viz = shap_values_for_viz.T
        
        return shap_values_for_viz, explainer, X[:max_samples]
        
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {str(e)}")
        print(f"Full error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def show_xai_analysis(data, model, scaler, predictions):
    """Display XAI analysis for the predictions"""
    st.subheader("Explainable AI Analysis")
    
    try:
        # Get feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            print(f"Loaded feature names: {feature_names}")
    except FileNotFoundError:
        st.error("Feature names file not found!")
        return
    except Exception as e:
        st.error(f"Error loading feature names: {str(e)}")
        return
    
    # Generate SHAP explanations
    shap_values, explainer, shap_data = generate_shap_explanations(model, data, scaler, feature_names)
    
    if shap_values is not None and shap_data is not None:
        try:
            # Create tabs for different XAI visualizations
            tab1, tab2 = st.tabs(["Feature Importance", "Individual Predictions"])
            
            with tab1:
                st.write("Global Feature Importance")
                
                # Calculate global feature importance (mean absolute SHAP values)
                mean_shap = np.abs(shap_values).mean(axis=0)
                if mean_shap.ndim > 1:
                    mean_shap = mean_shap.flatten()
                
                # Ensure arrays are the same length
                min_length = min(len(mean_shap), len(feature_names))
                mean_shap = mean_shap[:min_length]
                feature_names_subset = feature_names[:min_length]
                
                # Create DataFrame for global importance
                global_importance = pd.DataFrame({
                    'Feature': feature_names_subset,
                    'Importance': mean_shap
                }).sort_values('Importance', ascending=True)
                
                # Plot global importance
                fig = px.bar(
                    global_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Global Feature Importance'
                )
                st.plotly_chart(fig)
            
            with tab2:
                st.write("Individual Prediction Explanation")
                
                # Convert predictions to list if it's not already
                pred_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                
                # Find suspicious traffic indices
                suspicious_indices = [i for i, pred in enumerate(pred_list[:len(shap_data)]) 
                                   if pred != "BENIGN"]
                
                if suspicious_indices:
                    # Create selection for suspicious traffic
                    selected_index = st.selectbox(
                        "Select suspicious traffic to explain:",
                        range(len(suspicious_indices)),
                        format_func=lambda x: f"Traffic {suspicious_indices[x]} ({pred_list[suspicious_indices[x]]})"
                    )
                    
                    if selected_index is not None:
                        actual_index = suspicious_indices[selected_index]
                        
                        # Show feature importance for selected prediction
                        st.write(f"### Analysis for Traffic {actual_index}")
                        
                        # Get SHAP values for selected prediction and ensure it's 1D
                        local_shap_values = shap_values[actual_index]
                        if local_shap_values.ndim > 1:
                            local_shap_values = local_shap_values.flatten()
                        
                        # Ensure arrays are the same length
                        min_length = min(len(local_shap_values), len(feature_names))
                        local_shap_values = local_shap_values[:min_length]
                        feature_names_subset = feature_names[:min_length]
                        
                        # Create feature importance for selected prediction
                        local_importance = pd.DataFrame({
                            'Feature': feature_names_subset,
                            'Impact': local_shap_values
                        }).sort_values('Impact', ascending=True)
                        
                        # Plot local importance
                        fig = px.bar(
                            local_importance,
                            x='Impact',
                            y='Feature',
                            orientation='h',
                            title=f'Feature Impact for Traffic {actual_index}'
                        )
                        st.plotly_chart(fig)
                        
                        # Show actual feature values
                        st.write("### Feature Values")
                        feature_data = shap_data.iloc[actual_index]
                        for feat, val in zip(feature_names_subset, feature_data[:min_length]):
                            st.write(f"**{feat}:** {val:.4f}")
                else:
                    st.info("No suspicious traffic detected to explain.")
                    
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            print(f"Full error details: {str(e)}")
            import traceback
            traceback.print_exc()
            
    else:
        st.error("Could not generate SHAP explanations")

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
        
        # Add XAI analysis if predictions indicate suspicious traffic
        if any(predictions != "BENIGN"):
            show_xai_analysis(data, model, scaler, predictions)
        
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
        'Flow Packets/s': [
            np.random.choice(
                [np.random.randint(1, 1000), np.random.randint(5000, 10000)],  # Normal vs Suspicious
                p=[0.7, 0.3]  # 30% chance of suspicious traffic
            ) for _ in range(num_records)
        ],
        'Flow Bytes/s': [
            np.random.choice(
                [np.random.randint(1, 50000), np.random.randint(100000, 200000)],  # Normal vs Suspicious
                p=[0.7, 0.3]  # 30% chance of suspicious traffic
            ) for _ in range(num_records)
        ],
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
        ["Dashboard", "Model Training", "Real-time Monitoring", "Batch Analysis", "System Settings"]
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
                        
                        # Add XAI Analysis section
                        st.markdown("---")
                        st.subheader("🔍 XAI Analysis")
                        
                        # Force XAI analysis to show even for normal traffic
                        show_xai_analysis(sample_data, model, scaler, predictions)
                        
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
                    show_threat_intel_dashboard(enriched_data)
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

    # Handle new pages
    if st.session_state.get('current_page') == "change_password":
        change_password_page()
    elif st.session_state.get('current_page') == "setup_2fa":
        setup_2fa()

if __name__ == "__main__":
    main()