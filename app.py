# from flask import Flask,request,render_template
# import numpy as np
# import pandas
# import sklearn
# import pickle

# model = pickle.load(open('model.pkl','rb'))
# sc = pickle.load(open('standscaler.pkl','rb'))
# mx = pickle.load(open('minmaxscaler.pkl','rb'))


# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route("/predict",methods=['POST'])
# def predict():
#     N = request.form['Nitrogen']
#     P = request.form['Phosporus']
#     K = request.form['Potassium']
#     temp = request.form['Temperature']
#     humidity = request.form['Humidity']
#     ph = request.form['pH']
#     rainfall = request.form['Rainfall']

#     feature_list = [N, P, K, temp, humidity, ph, rainfall]
#     single_pred = np.array(feature_list).reshape(1, -1)

#     mx_features = mx.transform(single_pred)
#     sc_mx_features = sc.transform(mx_features)
#     prediction = model.predict(sc_mx_features)

#     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                  8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                  14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                  19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there".format(crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
#     return render_template('index.html',result = result)


# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
import numpy as np
import pickle

# ---------- Page config ----------
st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="centered")

# ---------- Load model & scalers ----------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    return model, sc, mx

model, sc, mx = load_artifacts()

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# ---------- UI ----------
st.title("🌾 Crop Recommendation System")
st.write("Enter your soil and climate parameters to get the best crop recommendation.")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=350.0, value=100.0, step=1.0)

st.markdown("---")

if st.button("🔍 Predict Best Crop", use_container_width=True):
    try:
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            st.success(f"🌱 **{crop}** is the best crop to be cultivated there.")
        else:
            st.error("Sorry, we could not determine the best crop with the provided data.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
#st.caption("Make sure `model.pkl`, `standscaler.pkl`, and `minmaxscaler.pkl` are in the same folder as this app.")