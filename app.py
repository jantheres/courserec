import streamlit as st
import pandas as pd
import pickle

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset1.csv')

# Features used for prediction
features = ['Goal', 'Interest']

# Load the best model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Streamlit app
st.title('Course Recommendation System')

# User inputs for goal and interest
goals = df['Goal'].unique()
interests = df['Interest'].unique()
selected_goal = st.selectbox('Select your goal:', goals)
selected_interest = st.selectbox('Select your interest:', interests)

if st.button('Get Recommendations'):
    # Create a new data frame with user inputs
    X_new = pd.DataFrame([[selected_goal, selected_interest]], columns=features)
    
    # Get prediction probabilities
    pred_probs = model.predict_proba(X_new)[0]
    
    # Get the top N recommendations (e.g., top 3)
    N = 3
    top_indices = pred_probs.argsort()[-N:][::-1]
    recommended_courses = le.inverse_transform(top_indices)
    
    # Display the recommendations
    st.write('Top Course Recommendations:')
    for i, course in enumerate(recommended_courses, start=1):
        st.write(f"{i}. {course}")