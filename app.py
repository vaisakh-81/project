# app.py
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Train and save the model
def train_and_save_model():
    # Load and prepare data (assuming insurance_dataset.csv exists)
    insurance_dataset = pd.read_csv('C:/Users/KELTRON/Desktop/pr/insurance (1).csv')
    
    # Encoding categorical variables
    insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)
    insurance_dataset.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
    insurance_dataset.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)
    
    # Split features and target
    X = insurance_dataset.drop(columns='charges', axis=1)
    y = insurance_dataset['charges']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save model using pickle
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load model
    model = load_model()
    
    # Get form data
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])
    
    # Create input array
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return render_template('output.html', prediction=prediction)

if __name__ == '__main__':
    # Train and save model if not exists
    import os
    if not os.path.exists('model.pkl'):
        train_and_save_model()
    app.run(debug=True)