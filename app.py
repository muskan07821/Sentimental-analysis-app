
import streamlit as st
from transformers import pipeline

# Title for the app
st.title("Sentiment Classification with BERT")

# Load the sentiment analysis model
# Replace with your model if fine-tuned (e.g., 'bert-base-uncased-sentiment-model')
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Input text area
text = st.text_area("Enter Your Text Here")

# Predict button
if st.button("Predict"):
    if text.strip():  # Check if text is not empty
        result = classifier(text)  # Perform prediction
        st.write("Prediction Result:", result)  # Display result
    else:
        st.write("Please enter some text for analysis.")
