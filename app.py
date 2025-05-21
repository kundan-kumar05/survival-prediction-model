import streamlit as st
import pandas as pd
import pickle

# Load the saved model using pickle
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app UI
st.title("🚢 Titanic Survival Prediction")
st.markdown("Fill in the passenger details:")

# Inputs from user
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ['male', 'female'])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare Paid", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Prepare input for model
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [0 if Sex == 'male' else 1],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [0 if Embarked == 'S' else 1 if Embarked == 'C' else 2]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Passenger Survived!")
    else:
        st.error("☠️ Passenger Did Not Survive.")
