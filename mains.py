import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data=pd.read_csv('diabetes_data.csv')
print(data) X = data.drop('Outcome', axis=1) 
Y = data['Outcome'] X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42) 
model = LogisticRegression(max_iter=1000) model.fit(X_train, Y_train) st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes.") 
input_data = [] for col in X.columns: value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1) 
input_data.append(value)
if st.button("Predict"): prediction = model.predict([input_data]) 
result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic" 
st.success(f"Prediction: {result}") 
st.subheader("Model Performance") 
Y_pred = model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y.test, y_pred))

