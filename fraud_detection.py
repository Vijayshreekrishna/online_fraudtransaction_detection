import streamlit as st
import joblib
import pandas as pd
from PIL import Image


st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🚨"
)

# Load the model
model = joblib.load('fraud_detection_pipeline.pkl')

st.title('Online Fraud Detection Application')



st.subheader('Please enter the transaction details below:')

st.divider()

transaction_type = st.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input('Old Balance (Sender)', min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input('New Balance (Sender)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
    'type': [transaction_type],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest]
})

    
    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction Result: '{int(prediction)}'")

    if prediction == 1:
        st.error('This transaction is likely fraud')
    else:
        st.success('This transaction is likely legitimate')
        
        
st.divider()
st.info('Note: The model is trained on synthetic data and may not reflect real-world scenarios accurately.')