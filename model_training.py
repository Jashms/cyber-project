import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    # Load the data
    df = pd.read_csv('friday.csv')
    
    # Drop any non-numeric columns except 'Label'
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    features = [col for col in numeric_cols if col != 'Label']
    
    # Prepare features and target
    X = df[features]
    y = df['Label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model, scaler, and feature names
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Return model accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    return train_accuracy, test_accuracy, features

if __name__ == "__main__":
    train_model() 