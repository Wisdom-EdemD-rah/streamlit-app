import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def preprocess_data(data):
    # Convert 'TotalCharges' to numeric, handling non-numeric values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Fill NaN values in 'TotalCharges' with the mean of the column
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

    # Encode categorical variables and create a mapping of the possible values
    encoder = LabelEncoder()
    categorical_features = ['SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']
    encoder_mapping = {}
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])
        encoder_mapping[feature] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data, encoder_mapping, scaler

def train_model(data):
    # Split the data into features and target
    X = data.drop(columns=['Churn', 'customerID'])
    y = data['Churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def save_model_and_encoder(model, encoder_mapping, scaler):
    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    joblib.dump(model, os.path.join('models', 'model.pkl'))

    # Save the encoder mapping
    joblib.dump(encoder_mapping, os.path.join('models', 'encoder_mapping.pkl'))

    # Save the scaler
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))

def main():
    # Load the dataset using an absolute path
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'final_test_set_churn.csv'))
    data = pd.read_csv(data_path)

    # Preprocess the data
    data, encoder_mapping, scaler = preprocess_data(data)

    # Train the model
    model = train_model(data)

    # Save the model and encoder mapping
    save_model_and_encoder(model, encoder_mapping, scaler)

if __name__ == "__main__":
    main()