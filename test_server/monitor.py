import psutil
import time
import pandas as pd
from datetime import datetime
import queue

class NetworkMonitor:
    def __init__(self):
        self.prev_bytes_sent = 0
        self.prev_bytes_recv = 0
        self.prev_time = time.time()
        
    def get_network_stats(self):
        # Get network statistics
        net_stats = psutil.net_io_counters()
        current_time = time.time()
        
        # Calculate bytes per second
        bytes_sent = net_stats.bytes_sent
        bytes_recv = net_stats.bytes_recv
        time_diff = current_time - self.prev_time
        
        bytes_sent_sec = (bytes_sent - self.prev_bytes_sent) / time_diff
        bytes_recv_sec = (bytes_recv - self.prev_bytes_recv) / time_diff
        
        # Update previous values
        self.prev_bytes_sent = bytes_sent
        self.prev_bytes_recv = bytes_recv
        self.prev_time = current_time
        
        # Create packet data matching your model features
        packet_data = {
            'Src IP dec': 127001,  # localhost
            'Src Port': 0,
            'Dst IP dec': 127001,
            'Dst Port': 5001,
            'Protocol': 6,  # TCP
            'Flow Duration': time_diff * 1000,  # in milliseconds
            'Total Fwd Packet': net_stats.packets_sent,
            'Total Bwd packets': net_stats.packets_recv,
            'Total Length of Fwd Packet': bytes_sent,
            'Total Length of Bwd Packet': bytes_recv,
            'Fwd Packet Length Max': bytes_sent_sec,
            'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': bytes_sent_sec/2,
            'Fwd Packet Length Std': 0,
            'Bwd Packet Length Max': bytes_recv_sec,
            'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': bytes_recv_sec/2,
            'Bwd Packet Length Std': 0,
            'Flow Bytes/s': bytes_sent_sec + bytes_recv_sec,
            'Flow Packets/s': (net_stats.packets_sent + net_stats.packets_recv) / time_diff,
            'Flow IAT Mean': time_diff * 1000,
            'Flow IAT Std': 0,
            'Flow IAT Max': time_diff * 1000,
            'Flow IAT Min': 0,
            'Fwd IAT Total': time_diff * 1000,
            'Fwd IAT Mean': time_diff * 1000,
            'Fwd IAT Std': 0,
            'Fwd IAT Max': time_diff * 1000,
            'Fwd IAT Min': 0,
            'Bwd IAT Total': time_diff * 1000,
            'Bwd IAT Mean': time_diff * 1000,
            'Bwd IAT Std': 0,
            'Bwd IAT Max': time_diff * 1000,
            'Bwd IAT Min': 0,
            'Fwd PSH Flags': 0,
            'Bwd PSH Flags': 0,
            'Fwd URG Flags': 0,
            'Bwd URG Flags': 0,
            'Fwd Header Length': 0,
            'Bwd Header Length': 0,
            'Fwd Packets/s': net_stats.packets_sent / time_diff,
            'Bwd Packets/s': net_stats.packets_recv / time_diff,
            'Packet Length Min': 0,
            'Packet Length Max': max(bytes_sent_sec, bytes_recv_sec),
            'Packet Length Mean': (bytes_sent_sec + bytes_recv_sec) / 2,
            'Packet Length Std': 0,
            'Packet Length Variance': 0,
            'FIN Flag Count': 0,
            'SYN Flag Count': 0,
            'RST Flag Count': 0,
            'PSH Flag Count': 0,
            'ACK Flag Count': 0,
            'URG Flag Count': 0,
            'CWR Flag Count': 0,
            'ECE Flag Count': 0,
            'Down/Up Ratio': bytes_recv_sec / bytes_sent_sec if bytes_sent_sec > 0 else 0,
            'Average Packet Size': (bytes_sent + bytes_recv) / (net_stats.packets_sent + net_stats.packets_recv) if (net_stats.packets_sent + net_stats.packets_recv) > 0 else 0,
            'Fwd Segment Size Avg': bytes_sent / net_stats.packets_sent if net_stats.packets_sent > 0 else 0,
            'Bwd Segment Size Avg': bytes_recv / net_stats.packets_recv if net_stats.packets_recv > 0 else 0,
            'Fwd Bytes/Bulk Avg': 0,
            'Fwd Packet/Bulk Avg': 0,
            'Fwd Bulk Rate Avg': 0,
            'Bwd Bytes/Bulk Avg': 0,
            'Bwd Packet/Bulk Avg': 0,
            'Bwd Bulk Rate Avg': 0,
            'Subflow Fwd Packets': net_stats.packets_sent,
            'Subflow Fwd Bytes': bytes_sent,
            'Subflow Bwd Packets': net_stats.packets_recv,
            'Subflow Bwd Bytes': bytes_recv,
            'FWD Init Win Bytes': 0,
            'Bwd Init Win Bytes': 0,
            'Fwd Act Data Pkts': net_stats.packets_sent,
            'Fwd Seg Size Min': 0,
            'Active Mean': time_diff * 1000,
            'Active Std': 0,
            'Active Max': time_diff * 1000,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0
        }
        
        return packet_data

def start_monitoring(callback):
    monitor = NetworkMonitor()
    while True:
        stats = monitor.get_network_stats()
        callback(stats)
        time.sleep(1)

if __name__ == "__main__":
    def print_stats(stats):
        print(f"Network stats: {stats}")
    
    start_monitoring(print_stats) 