# 🌾 Crop Recommendation System

A machine learning web application that recommends the most suitable crop to cultivate based on soil nutrient levels and climate conditions. Built with Streamlit and scikit-learn.

## Overview

Given seven input parameters — Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall — the model predicts the best crop to grow from a set of 22 possible crops, including rice, maize, cotton, fruits (apple, banana, mango, grapes, etc.), and pulses (lentil, chickpea, kidney beans, etc.).

## Features

- Simple, interactive web interface for entering soil and climate data
- Trained `RandomForestClassifier` model served via a Streamlit app
- Feature scaling pipeline (MinMaxScaler + StandardScaler) applied before prediction
- Instant crop recommendation displayed on the same page
- Model and scalers cached with `st.cache_resource` for fast repeated predictions

## Tech Stack

- **App framework:** Python, Streamlit
- **Machine Learning:** scikit-learn, pandas, numpy
- **Model exploration:** Jupyter Notebook (`Crop_Recommendation_Using_Machine_Learning.ipynb`)

## Project Structure

```
├── app.py                                          # Streamlit application
├── model.pkl                                       # Trained RandomForestClassifier model
├── standscaler.pkl                                 # Fitted StandardScaler
├── minmaxscaler.pkl                                # Fitted MinMaxScaler
├── Crop_Recommendation_Using_Machine_Learning.ipynb # Model training notebook
├── requirements.txt
└── README.md
```

## Input Parameters

| Field | Description | Unit |
|---|---|---|
| Nitrogen | Nitrogen content in soil | mg/kg |
| Phosphorus | Phosphorus content in soil | mg/kg |
| Potassium | Potassium content in soil | mg/kg |
| Temperature | Ambient temperature | °C |
| Humidity | Relative humidity | % |
| pH | Soil pH value | — |
| Rainfall | Rainfall amount | mm |

## Supported Crops

Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee.

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd crop-recommendation-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   Make sure `model.pkl`, `standscaler.pkl`, and `minmaxscaler.pkl` are in the project root. If missing, run the training notebook (`Crop_Recommendation_Using_Machine_Learning.ipynb`) to regenerate them.

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Your browser should open automatically at:
   ```
   http://localhost:8501
   ```

3. Enter the soil nutrient and climate values in the form and click **🔍 Predict Best Crop** to get a recommendation.

## Model Training

The notebook `Crop_Recommendation_Using_Machine_Learning.ipynb` covers:
- Exploratory data analysis (distribution plots, correlation heatmap)
- Feature scaling with `MinMaxScaler` and `StandardScaler`
- Training and comparing multiple classifiers (Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, Random Forest, Bagging, Gradient Boosting, AdaBoost)
- Selecting `RandomForestClassifier` as the best-performing model and exporting it along with the fitted scalers using `pickle`

## Future Improvements

- Add input validation and error handling for non-numeric or out-of-range values
- Display model confidence/probability alongside the prediction
- Containerize the app with Docker for easier deployment
- Add unit tests for the prediction pipeline
- Re-fit `mx`/`sc` scalers with `.transform()` instead of `.fit_transform()` at inference time in the notebook's `recommendation()` helper, so scaling parameters aren't refit on single-sample inputs

## License

