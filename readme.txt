# Patient Visit Frequency Predictor

## Project Overview

This project is designed to predict the recommended frequency of patient visits based on various clinical and risk factor inputs. The project includes three main components:
1. **Federated Learning Model Training**: Uses federated learning to train a neural network model.
2. **Streamlit Frontend**: A web application for inputting patient data and visualizing predictions.
3. **FastAPI Backend**: A backend server to handle prediction requests and responses.

## Components

### 1. Federated Learning Model Training

**File**: `FLmodel.py`

This component involves training a neural network model using federated learning. Key steps include:
- Load and preprocess the dataset.
- Define a simple neural network using PyTorch.
- Implement federated learning by distributing data among clients and aggregating their model updates.

**Main Libraries Used**:
- `scikit-learn` for data preprocessing.
- `pandas` for data manipulation.
- `torch` for building and training neural networks.
- `matplotlib` for plotting results.

**Setup and Usage**:
1. Ensure `chronickidneydiseases.csv` is in the same directory as the script.
2. Install the necessary libraries:
    ```sh
    pip install scikit-learn pandas torch matplotlib
    ```
3. Run the training script:
    ```sh
    python FLmodel.py
    ```
4. The trained model will be saved as `federated_model.pth`.

### 2. Streamlit Frontend

**File**: `frontend.py`

This Streamlit web application allows users to input patient clinical data and get predictions on the recommended number of visits per month.

**Main Libraries Used**:
- `streamlit` for building the web application.
- `pandas` for data handling.
- `requests` for making HTTP requests to the backend.
- `numpy` for numerical operations.

**Setup and Usage**:
1. Install Streamlit and other dependencies:
    ```sh
    pip install streamlit pandas requests numpy
    ```
2. Run the Streamlit application:
    ```sh
    streamlit run frontend.py
    ```
3. Enter the patient information in the web form and click the "Predict Recommended Visits" button to get the prediction.

### 3. FastAPI Backend

**File**: `backend.py`

This FastAPI backend server handles prediction requests. It receives patient data, processes it, and returns the predicted number of visits.

**Main Libraries Used**:
- `fastapi` for building the backend API.
- `pydantic` for data validation.
- `numpy` for numerical operations.
- `uvicorn` for running the server.

**Setup and Usage**:
1. Install FastAPI and Uvicorn:
    ```sh
    pip install fastapi pydantic numpy uvicorn
    ```
2. Run the backend server:
    ```sh
    python backend.py
    ```
3. The backend will be accessible at `http://localhost:8001`.

## Detailed Instructions

### Setting Up the Environment

1. **Create a Virtual Environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

1. **Start the Backend Server**:
    ```sh
    python backend.py
    ```

2. **Run the Frontend Application**:
    ```sh
    streamlit run frontend.py
    ```

3. **Train the Federated Model** (if needed):
    ```sh
    python FLmodel.py
    ```

### Making Predictions

To make a prediction, follow these steps:
1. Open the Streamlit web application.
2. Enter the required patient information.
3. Click the "Predict Recommended Visits" button.
4. The prediction result will be displayed on the web page.

## Project Structure

patient-visit-predictor/
│
├── FLmodel.py                  # Federated learning training script
├── frontend.py                 # Streamlit frontend application
├── backend.py                  # FastAPI backend server
├── chronickidneydiseases.csv   # Dataset file
└── requirements.txt            # Dependencies
