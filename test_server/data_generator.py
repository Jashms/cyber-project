import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TrafficDataGenerator:
    def __init__(self):
        self.base_time = datetime.now()
    
    def generate_normal_traffic(self):
        return {
            'Src IP dec': np.random.randint(1000000000, 9999999999),
            'Src Port': np.random.randint(1024, 65535),
            'Dst IP dec': np.random.randint(1000000000, 9999999999),
            'Dst Port': np.random.randint(1, 1024),  # Common service ports
            'Protocol': np.random.choice([6, 17]),  # TCP or UDP
            'Flow Duration': np.random.randint(100, 1000),
            'Total Fwd Packet': np.random.randint(1, 10),
            'Total Bwd packets': np.random.randint(1, 10),
            'Total Length of Fwd Packet': np.random.randint(100, 1500),
            'Total Length of Bwd Packet': np.random.randint(100, 1500),
            'Fwd Packet Length Max': np.random.randint(100, 1500),
            'Fwd Packet Length Min': np.random.randint(40, 100),
            'Fwd Packet Length Mean': np.random.randint(100, 800),
            'Fwd Packet Length Std': np.random.random() * 100,
            'Bwd Packet Length Max': np.random.randint(100, 1500),
            'Bwd Packet Length Min': np.random.randint(40, 100),
            'Bwd Packet Length Mean': np.random.randint(100, 800),
            'Bwd Packet Length Std': np.random.random() * 100,
            'Flow Bytes/s': np.random.random() * 1000,
            'Flow Packets/s': np.random.random() * 100,
            'Flow IAT Mean': np.random.random() * 100,
            'Flow IAT Std': np.random.random() * 50,
            'Flow IAT Max': np.random.random() * 200,
            'Flow IAT Min': np.random.random() * 10,
            'Fwd IAT Total': np.random.random() * 1000,
            'Fwd IAT Mean': np.random.random() * 100,
            'Fwd IAT Std': np.random.random() * 50,
            'Fwd IAT Max': np.random.random() * 200,
            'Fwd IAT Min': np.random.random() * 10,
            'Bwd IAT Total': np.random.random() * 1000,
            'Bwd IAT Mean': np.random.random() * 100,
            'Bwd IAT Std': np.random.random() * 50,
            'Bwd IAT Max': np.random.random() * 200,
            'Bwd IAT Min': np.random.random() * 10,
            'Fwd PSH Flags': 0,
            'Bwd PSH Flags': 0,
            'Fwd URG Flags': 0,
            'Bwd URG Flags': 0,
            'Fwd Header Length': np.random.randint(20, 40),
            'Bwd Header Length': np.random.randint(20, 40),
            'Fwd Packets/s': np.random.random() * 50,
            'Bwd Packets/s': np.random.random() * 50,
            'Packet Length Min': np.random.randint(40, 100),
            'Packet Length Max': np.random.randint(100, 1500),
            'Packet Length Mean': np.random.randint(100, 800),
            'Packet Length Std': np.random.random() * 100,
            'Packet Length Variance': np.random.random() * 10000,
            'FIN Flag Count': 0,
            'SYN Flag Count': 0,
            'RST Flag Count': 0,
            'PSH Flag Count': 0,
            'ACK Flag Count': 1,
            'URG Flag Count': 0,
            'CWR Flag Count': 0,
            'ECE Flag Count': 0,
            'Down/Up Ratio': np.random.random() * 2,
            'Average Packet Size': np.random.randint(100, 800),
            'Fwd Segment Size Avg': np.random.randint(100, 800),
            'Bwd Segment Size Avg': np.random.randint(100, 800),
            'Fwd Bytes/Bulk Avg': 0,
            'Fwd Packet/Bulk Avg': 0,
            'Fwd Bulk Rate Avg': 0,
            'Bwd Bytes/Bulk Avg': 0,
            'Bwd Packet/Bulk Avg': 0,
            'Bwd Bulk Rate Avg': 0,
            'Subflow Fwd Packets': np.random.randint(1, 10),
            'Subflow Fwd Bytes': np.random.randint(100, 1500),
            'Subflow Bwd Packets': np.random.randint(1, 10),
            'Subflow Bwd Bytes': np.random.randint(100, 1500),
            'FWD Init Win Bytes': np.random.randint(1000, 65535),
            'Bwd Init Win Bytes': np.random.randint(1000, 65535),
            'Fwd Act Data Pkts': np.random.randint(1, 10),
            'Fwd Seg Size Min': np.random.randint(40, 100),
            'Active Mean': np.random.random() * 100,
            'Active Std': np.random.random() * 50,
            'Active Max': np.random.random() * 200,
            'Active Min': np.random.random() * 10,
            'Idle Mean': np.random.random() * 100,
            'Idle Std': np.random.random() * 50,
            'Idle Max': np.random.random() * 200,
            'Idle Min': np.random.random() * 10
        }

    def generate_attack_traffic(self, attack_type="DOS"):
        base_traffic = self.generate_normal_traffic()
        
        if attack_type == "DOS":
            # More extreme values for DOS attack
            base_traffic.update({
                'Flow Packets/s': np.random.randint(5000, 10000),  # Much higher packet rate
                'Flow Bytes/s': np.random.randint(1000000, 5000000),  # Much higher byte rate
                'Total Fwd Packet': np.random.randint(500, 1000),  # More forward packets
                'Total Bwd packets': np.random.randint(1, 5),
                'Flow IAT Mean': np.random.random() * 0.01,  # Very small inter-arrival time
                'Flow IAT Std': np.random.random() * 0.01,
                'Fwd Packets/s': np.random.randint(5000, 10000),  # High forward packet rate
                'Packet Length Variance': np.random.random() * 100,  # Low variance in packet size
                'PSH Flag Count': np.random.randint(10, 50),  # High number of PSH flags
                'SYN Flag Count': np.random.randint(100, 500)  # High number of SYN flags for DOS
            })
        elif attack_type == "Port Scan":
            # More extreme values for Port Scan
            base_traffic.update({
                'Dst Port': np.random.randint(1, 65535),
                'Flow Duration': np.random.randint(1, 50),  # Very short flows
                'Total Fwd Packet': np.random.randint(1, 3),  # Few packets per flow
                'Total Bwd packets': np.random.randint(0, 2),
                'Flow Packets/s': np.random.randint(100, 500),  # High packet rate
                'SYN Flag Count': np.random.randint(1, 3),  # SYN flags present
                'Flow Bytes/s': np.random.randint(50, 200),  # Small amount of data
                'Fwd Packet Length Mean': np.random.randint(40, 60),  # Small packets
                'Packet Length Variance': np.random.random() * 10  # Very consistent packet sizes
            })
        
        return base_traffic

    def generate_dataset(self, num_records, attack_ratio=0.2, attack_types=["DOS", "Port Scan"]):
        data = []
        
        num_attack_records = int(num_records * attack_ratio)
        num_normal_records = num_records - num_attack_records
        
        # Generate normal traffic
        for _ in range(num_normal_records):
            record = self.generate_normal_traffic()
            record['Label'] = 'BENIGN'
            data.append(record)
        
        # Generate attack traffic
        for _ in range(num_attack_records):
            attack_type = np.random.choice(attack_types)
            record = self.generate_attack_traffic(attack_type)
            record['Label'] = attack_type
            data.append(record)
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)

def generate_test_data(num_records, attack_ratio, attack_types):
    # Create test_data directory if it doesn't exist
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    
    # Generate data
    generator = TrafficDataGenerator()
    df = generator.generate_dataset(num_records, attack_ratio, attack_types)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'test_data/traffic_data_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    return filename

if __name__ == "__main__":
    # Test the generator
    filename = generate_test_data(1000, 0.2, ["DOS", "Port Scan"])
    print(f"Generated test data: {filename}") 