import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# load th tensorflow model for prediction

model=load_model('model.h5')

with open('tokenizer.pkl', 'rb') as r:
    tokenizer=pickle.load(r)


st.title("Twitter tweets Sentiment analysis")

tweet=st.text_area("Enter the tweet :")

if st.button("Prdict Sentiment") and tweet.strip(): 
    sequences=tokenizer.texts_to_sequences([tweet])
    sequences=pad_sequences(sequences, padding='post', maxlen=166)

    prediction=model.predict(sequences)

    predicted_class=np.argmax(prediction, axis=1)[0]

    sentiment_map={0:'Negative', 1:'Neutral', 2:'Positive'}

    st.write("Sentiment :" ,sentiment_map[predicted_class])