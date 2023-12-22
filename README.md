# Machine-Learning-Project-Intrusion-Network-Detection

## Overview
The dataset to be audited was provided which consists of a wide variety of intrusions simulated in a military network environment. It created an environment to acquire raw TCP/IP dump data for a network by simulating a typical US Air Force LAN. The LAN was focused like a real environment and blasted with multiple attacks. A connection is a sequence of TCP packets starting and ending at some time duration between which data flows to and from a source IP address to a target IP address under some well-defined protocol. Also, each connection is labeled as either normal or as an attack with exactly one specific attack type. Each connection record consists of about 100 bytes. For each TCP/IP connection, 41 quantitative and qualitative features are obtained from normal and attack data (3 qualitative and 38 quantitative features).
The class variable has two categories: 
• Normal 
• Anomalous

## Prerequisites
- Python
- Flask
- Numpy
- Pickle

## How to Use
1. Install the required dependencies:
    ```bash
    pip install flask numpy
    ```
2. Run the Flask app:
    ```bash
    python app.py
    ```
3. Access the web interface at `http://localhost:5000` in your browser.

## Files
- `model.pkl`: Pickled machine learning model for class prediction.
- `app.py`: Flask web application script.

## Web Interface
- **Home Page (`/`)**: Default page with a link to the prediction form.
- **Prediction Form (`/predict`)**: Form to input features and get class predictions.

## How to Predict
1. Visit the home page (`http://localhost:5000`).
2. Click on the link to the prediction form.
3. Input values for the features.
4. Click the "Predict" button.
5. View the predicted class on the results page.

## Model Details
- **Model Type**: Machine learning model (specific details not provided in the code).
- **Model Loading**: The model is loaded from the `model.pkl` file.

## Flask Routes
- `/`: Home page with a link to the prediction form.
- `/predict`: Route for predicting the class based on user input.

## Usage Notes
- Ensure that the necessary dependencies are installed before running the Flask app.
- The web app runs on `http://localhost:5000` by default.

Feel free to customize this README according to your specific needs and provide any additional details that may be helpful for users. Include information about the model used, features required for prediction, and any special considerations.
