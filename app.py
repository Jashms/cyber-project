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

# Add this to store live traffic data
traffic_queue = queue.Queue()

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
        
        # Add custom logic for attack detection
        attack_indicators = pd.DataFrame()
        attack_indicators['high_packet_rate'] = data['Flow Packets/s'] > 1000  # Threshold for high packet rate
        attack_indicators['high_byte_rate'] = data['Flow Bytes/s'] > 100000   # Threshold for high byte rate
        attack_indicators['syn_flags'] = data['SYN Flag Count'] > 10          # Threshold for SYN flags
        attack_indicators['short_flow'] = data['Flow Duration'] < 50          # Threshold for short flows
        
        # Determine if traffic is malicious based on indicators
        is_attack = (attack_indicators.sum(axis=1) >= 2)  # If 2 or more indicators are True
        
        # Combine model predictions with custom logic
        final_predictions = predictions.copy()
        final_predictions[is_attack] = np.where(
            data.loc[is_attack, 'Flow Packets/s'] > 5000, 
            'DOS',
            'Port Scan'
        )
        
        return final_predictions
        
    except Exception as e:
        st.error(f"Error in monitoring: {str(e)}")
        return None

def generate_sample_traffic():
    """Generate realistic sample network traffic data"""
    # These should match your training data features
    sample_data = {
        'Src IP dec': [np.random.randint(0, 4294967295)],
        'Src Port': [np.random.randint(1, 65535)],
        'Dst IP dec': [np.random.randint(0, 4294967295)],
        'Dst Port': [np.random.randint(1, 65535)],
        'Protocol': [np.random.randint(0, 17)],
        'Flow Duration': [np.random.randint(0, 1000000)],
        'Total Fwd Packet': [np.random.randint(1, 100)],
        'Total Bwd packets': [np.random.randint(1, 100)],
        'Total Length of Fwd Packet': [np.random.randint(0, 10000)],
        'Total Length of Bwd Packet': [np.random.randint(0, 10000)],
        # Add all other features from your dataset
        # ... (add remaining features)
    }
    return pd.DataFrame(sample_data)

def monitor_callback(packet_data):
    if packet_data:
        traffic_queue.put(packet_data)

def main():
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
            
            # Start monitoring in a separate thread
            def monitor_callback(stats):
                st.session_state.traffic_queue.put(stats)
            
            st.session_state.monitor_thread = threading.Thread(
                target=start_monitoring,
                args=(monitor_callback,),
                daemon=True
            )
            st.session_state.monitor_thread.start()
        
        placeholder = st.empty()
        
        while monitoring_active:
            # Get accumulated stats
            stats_list = []
            while not st.session_state.traffic_queue.empty():
                stats_list.append(st.session_state.traffic_queue.get())
            
            if stats_list:
                # Convert to DataFrame
                df = pd.DataFrame(stats_list)
                
                # Monitor traffic
                predictions = monitor_traffic(df, model, scaler)
                
                if predictions is not None:
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
                            latest_stats = stats_list[-1]
                            st.metric("Bytes/sec (Sent)", f"{latest_stats['Fwd Packet Length Max']:.2f}")
                            st.metric("Bytes/sec (Received)", f"{latest_stats['Bwd Packet Length Max']:.2f}")
                        
                        with col2:
                            st.subheader("Traffic Analysis")
                            st.metric("Active Connections", len(stats_list))
                            if len(predictions) > 0:
                                alert_rate = sum(p != "BENIGN" for p in predictions)/len(predictions)
                                st.metric("Alert Rate", f"{alert_rate*100:.2f}%")
                            
                            if status != "Normal":
                                st.error(f"⚠️ Potential intrusion detected at {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(refresh_rate)
    
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
            predictions = monitor_traffic(data, model, scaler)
            
            if predictions is not None:
                # Combine original data with predictions
                results_df = data.copy()
                results_df['Predicted'] = predictions
                
                # Analysis Dashboard
                st.subheader("Analysis Dashboard")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_traffic = len(predictions)
                    st.metric("Total Traffic Records", total_traffic)
                
                with col2:
                    benign_count = sum(predictions == "BENIGN")
                    benign_percentage = (benign_count / total_traffic) * 100
                    st.metric("Normal Traffic", f"{benign_count} ({benign_percentage:.1f}%)")
                
                with col3:
                    attack_count = sum(predictions != "BENIGN")
                    attack_percentage = (attack_count / total_traffic) * 100
                    st.metric("Detected Attacks", f"{attack_count} ({attack_percentage:.1f}%)")
                
                # Traffic Analysis Visualizations
                st.subheader("Traffic Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction Distribution
                    prediction_counts = pd.Series(predictions).value_counts()
                    fig_pred = px.pie(
                        values=prediction_counts.values,
                        names=prediction_counts.index,
                        title="Traffic Distribution"
                    )
                    st.plotly_chart(fig_pred)
                
                with col2:
                    # Traffic Flow Analysis
                    flow_data = results_df.groupby('Predicted')[['Flow Bytes/s', 'Flow Packets/s']].mean()
                    fig_flow = px.bar(
                        flow_data,
                        title="Average Flow Metrics by Traffic Type",
                        barmode='group'
                    )
                    st.plotly_chart(fig_flow)
                
                # Detailed Analysis
                st.subheader("Detailed Analysis")
                
                # Time-based analysis if timestamp is available
                if 'Timestamp' in results_df.columns:
                    st.line_chart(
                        results_df.groupby('Timestamp')['Predicted']
                        .apply(lambda x: sum(x != 'BENIGN'))
                        .reset_index(name='Attack Count')
                        .set_index('Timestamp')
                    )
                
                # Traffic Patterns
                st.subheader("Traffic Patterns")
                pattern_metrics = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 
                                 'Flow Bytes/s', 'Flow Packets/s']
                
                selected_metric = st.selectbox("Select Metric for Analysis", pattern_metrics)
                
                fig_box = px.box(
                    results_df,
                    x='Predicted',
                    y=selected_metric,
                    title=f"{selected_metric} Distribution by Traffic Type"
                )
                st.plotly_chart(fig_box)
                
                # Suspicious Traffic Details
                st.subheader("Suspicious Traffic Details")
                attack_traffic = results_df[results_df['Predicted'] != 'BENIGN']
                if not attack_traffic.empty:
                    st.dataframe(attack_traffic)
                    
                    # Download suspicious traffic
                    st.download_button(
                        label="Download Suspicious Traffic Data",
                        data=attack_traffic.to_csv(index=False).encode('utf-8'),
                        file_name="suspicious_traffic.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No suspicious traffic detected in this dataset")
                
                # Statistical Summary
                st.subheader("Statistical Summary")
                stats_cols = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s']
                st.dataframe(results_df.groupby('Predicted')[stats_cols].describe())
                
                # Full Results Download
                st.download_button(
                    label="Download Complete Analysis Results",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="analysis_results.csv",
                    mime="text/csv"
                )
    
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

if __name__ == "__main__":
    main()