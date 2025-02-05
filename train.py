from model_training_dl import train_deep_model
import os

def main():
    # Check if Friday.csv exists
    if not os.path.exists('Friday.csv'):
        print("Error: Friday.csv not found! Please ensure the dataset is in the project directory.")
        return
    
    print("Starting model training...")
    accuracy, _, features = train_deep_model()
    
    if accuracy is not None:
        print(f"\nTraining completed successfully!")
        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"\nNumber of features used: {len(features)}")
    else:
        print("Training failed. Check the error messages above.")

if __name__ == "__main__":
    main() 