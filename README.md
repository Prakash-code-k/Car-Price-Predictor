# Car Price Prediction System

A machine learning–based web application that predicts the selling price of used cars using historical data and user inputs. The application is built using Python, Scikit-learn, and Streamlit.

## Project Overview

Buying or selling a used car often involves uncertainty in pricing. This project aims to predict the fair selling price of a used car based on important features such as year, fuel type, transmission, ownership, and kilometers driven using machine learning techniques.

## Features

- Predicts used car selling price
- Simple and interactive Streamlit interface
- Multiple machine learning models compared
- Fast prediction using pre-trained model
- Clean UI with background image

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- pickle

## Project Structure

Car-Price-Predictor/
│
├── app.py # Streamlit web app
├── model_training.ipynb # Model training & evaluation
├── best_model.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── dataset/
│ └── Car_details_v3_CLEANED # Dataset
│
└── assets/
|── background.jpg # Background image
└──background.mp4 # Background video

---

## Dataset Information

- Downloaded from kaggle and Cleaned

### Target Variable

- Selling Price

### Important Features

- Year
- Kms Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Engine
- Max Power
- Seats
- Mileage Value
- Mileage Unit

## Machine Learning Models Used

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boost Regressor
  The best-performing model is saved as `best_model.pkl` and used in the application.

## Install required dependencies:

`bash`

- pip install -r requirements.txt

## Run the Application

- streamlit run app.py

## Usage

- Enter car details such as year, fuel type, transmission, and kilometers driven...
- Click on Predict
- The predicted selling price will be displayed instantly
