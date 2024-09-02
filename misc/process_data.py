import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data_path = os.path.abspath('../data/data.csv')  # Adjust the path as needed
data = pd.read_csv(data_path)

# Identify the target variable
X = data.drop('left', axis=1)  # Features
y = data['left']  # Target variable

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Create a pipeline that first preprocesses the data and then fits the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model pipeline
joblib.dump(model_pipeline, '../models/model.pkl')

# Save the encoder separately
encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
joblib.dump(encoder, '../models/encoder.pkl')

# Save the scaler separately
scaler = model_pipeline.named_steps['preprocessor'].named_transformers_['num']
joblib.dump(scaler, '../models/scaler.pkl')

print("Model, encoder, and scaler saved successfully!")