import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def preprocess_friday_data():
    """Preprocess the Friday dataset"""
    print("Loading Friday dataset...")
    try:
        # Load the Friday dataset
        data = pd.read_csv('Friday.csv')
        
        # Select relevant features based on actual column names
        features = [
            'Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
            'Total Fwd Packet', 'Total Bwd packets',
            'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
            'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
            'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count',
            'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
            'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
            'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
            'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
            'Label'
        ]
        
        print(f"Selecting {len(features)} features from the dataset...")
        
        # Select available features
        available_features = [f for f in features if f in data.columns]
        data = data[available_features]
        
        # Handle missing values
        data = data.fillna(0)
        
        # Convert infinite values to 0
        data = data.replace([np.inf, -np.inf], 0)
        
        # Print unique labels before encoding
        print("\nUnique labels in dataset:", data['Label'].unique())
        
        # Convert label to binary (attack/normal)
        label_encoder = LabelEncoder()
        data['Label'] = label_encoder.fit_transform(data['Label'])
        
        # Print class distribution
        normal_count = len(data[data['Label'] == 0])
        attack_count = len(data[data['Label'] == 1])
        print(f"\nClass distribution after encoding:")
        print(f"Normal samples: {normal_count}")
        print(f"Attack samples: {attack_count}")
        print(f"Total samples: {len(data)}")
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Verify data is clean
        assert not data.isnull().any().any(), "Dataset contains null values"
        assert not (data.abs() > 1e9).any().any(), "Dataset contains extreme values"
        
        print(f"\nSuccessfully preprocessed {len(available_features)} features")
        return data
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print("Full error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()
        return None

def create_deep_model(input_shape):
    """Create a more efficient deep learning model"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Simplified architecture
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Optimizer with slightly higher learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.002)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )
    
    return model

def train_deep_model():
    """Train the model with reduced epochs"""
    try:
        # Preprocess data
        data = preprocess_friday_data()
        if data is None:
            return None, None, None
        
        # Prepare features and labels
        X = data.drop('Label', axis=1)
        y = data['Label']
        
        # Print unique label values after encoding
        print("\nUnique encoded labels:", np.unique(y))
        
        # Create class weights dictionary based on actual encoded values
        label_counts = y.value_counts()
        total_samples = len(y)
        class_weights = {
            label: total_samples / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
        print("\nClass weights:", class_weights)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modify the model for multi-class classification
        input_shape = X_train.shape[1]
        num_classes = len(np.unique(y))
        
        # Create model with correct output shape
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            # Output layer with correct number of neurons
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use legacy optimizer
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.002)
        
        # Compile model for multi-class
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel summary:")
        model.summary()
        
        # Add callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        # Train model
        print("\nStarting training...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model on test set...")
        test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"\nTest accuracy: {test_results[1]:.4f}")
        
        # Save model and related files
        print("\nSaving model and related files...")
        model.save('deep_model.h5')
        with open('deep_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('deep_feature_names.pkl', 'wb') as f:
            pickle.dump(list(X_train.columns), f)
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        return test_results[1], None, list(X_train.columns)
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict_deep_model(data):
    """Make predictions using the deep learning model"""
    try:
        # Load model and scaler
        model = models.load_model('deep_model.h5')
        with open('deep_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale features
        X_scaled = scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        return binary_predictions, predictions  # Return both binary predictions and probabilities
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

class DeepIDS:
    """Deep Learning-based Intrusion Detection System"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = models.load_model('deep_model.h5')
            with open('deep_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('deep_feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def predict(self, data):
        """Make predictions on new data"""
        if self.model is None:
            print("Model not loaded")
            return None
        
        try:
            # Ensure data has all required features
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0
            
            # Select and order features
            X = data[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None 