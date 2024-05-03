import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns

# Load the diamond dataset
# @st.cache
# def load_data():
#     return pd.read_csv("diamonds.csv")

diamonds_df = sns.load_dataset("diamonds")

# Preprocess the data
label_encoder = LabelEncoder()
diamonds_df['cut'] = label_encoder.fit_transform(diamonds_df['cut'])
diamonds_df['color'] = label_encoder.fit_transform(diamonds_df['color'])
diamonds_df['clarity'] = label_encoder.fit_transform(diamonds_df['clarity'])

# Train the model
X = diamonds_df[['carat', 'cut', 'color', 'clarity']]
y = diamonds_df['price']
model = RandomForestRegressor()
model.fit(X, y)

# Save the model as a pickle file
with open("diamond_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit app
st.title("Diamond Price Prediction")

# Sidebar inputs
carat = st.sidebar.slider("Carat", min_value=0.2, max_value=5.01, value=2.5, step=0.01)
cut = st.sidebar.selectbox("Cut", options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox("Color", options=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.sidebar.selectbox("Clarity", options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Convert categorical features to numerical
cut_encoded = label_encoder.transform([cut])[0]
color_encoded = label_encoder.transform([color])[0]
clarity_encoded = label_encoder.transform([clarity])[0]

# Make prediction
input_data = pd.DataFrame({'carat': [carat], 'cut': [cut_encoded], 'color': [color_encoded], 'clarity': [clarity_encoded]})
predicted_price = model.predict(input_data)[0]

# Display prediction
st.write("Predicted Price:", predicted_price)
