import streamlit as st 
from helper import *



st.header("NLP Final Project")
st.image('./img.png')
st.write("##### Enter text below to get the results")

input = st.text_area(label='', placeholder="Enter text here")

st.subheader("Choose a Task To Execute:")

option = st.radio('##### Options:', ('Named Entity Recognition', 'Sentiment Analysis', 'Text Summarization', 'Machine Translation'))





if option == 'Named Entity Recognition':
    ner(input)
if option == 'Sentiment Analysis': 
    st.subheader('Sentiment Analysis:')
    sentiment_analysis(input)
if option == 'Text Summarization':
    st.subheader('Text Summarization:')
    summarize(input)
if option == 'Machine Translation':
        st.subheader('Translation:')
        translate(input)
