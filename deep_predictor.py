import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model

class DeepPredictor:
    def __init__(self):
        # Load the trained model and preprocessing components
        self.model = load_model('deep_model.h5')
        
        # Load the scaler
        with open('deep_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load feature names
        with open('deep_feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
            
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def predict(self, data):
        """
        Make predictions on new network traffic data
        """
        try:
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0
            
            # Select and order features
            X = data[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Convert numeric predictions back to labels
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            
            # Calculate confidence scores
            confidence_scores = np.max(predictions, axis=1)
            
            return predicted_labels, confidence_scores
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, None 