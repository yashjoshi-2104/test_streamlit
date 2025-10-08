# app.py
# Real Estate Price Predictor using Streamlit
# Dependencies are listed in requirements.txt for Streamlit Cloud deployment

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@st.cache_data
def train_model():
    np.random.seed(42)
    num_samples = 1000
    
    locations = ['Downtown', 'Suburb', 'Rural']
    
    data = pd.DataFrame({
        'SquareFootage': np.random.randint(800, 4000, num_samples),
        'Bedrooms': np.random.randint(1, 6, num_samples),
        'Bathrooms': np.random.randint(1, 5, num_samples),
        'Location': np.random.choice(locations, num_samples)
    })
    
    base_price = 100000
    price = (base_price + 
             data['SquareFootage'] * 150 + 
             data['Bedrooms'] * 50000 + 
             data['Bathrooms'] * 30000 +
             data['Location'].map({'Downtown': 100000, 'Suburb': 50000, 'Rural': -20000}) +
             np.random.normal(0, 25000, num_samples))
             
    data['Price'] = price
    
    X = pd.get_dummies(data.drop('Price', axis=1), columns=['Location'], drop_first=True)
    y = data['Price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

model, trained_columns = train_model()

st.title('🏡 Real Estate Price Predictor')
st.write("Use the sliders and dropdown in the sidebar to get a price prediction for a property.")

st.sidebar.header('Input Property Features')

def user_input_features():
    sqft = st.sidebar.slider('Square Footage', 800, 4000, 1500)
    bedrooms = st.sidebar.slider('Bedrooms', 1, 5, 3)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 5, 2)
    location = st.sidebar.selectbox('Location', ('Downtown', 'Suburb', 'Rural'))
    
    data = {
        'SquareFootage': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location': location
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

input_processed = pd.get_dummies(input_df, columns=['Location'])
input_processed = input_processed.reindex(columns=trained_columns, fill_value=0)

prediction = model.predict(input_processed)

st.subheader('Prediction')
st.header(f'Estimated Price: ${prediction[0]:,.2f}')

st.write("---")
st.subheader('Your Input:')
st.dataframe(input_df)
