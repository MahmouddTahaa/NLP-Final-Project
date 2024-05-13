from transformers import MarianTokenizer, MarianMTModel, pipeline
import spacy 
import spacy_streamlit
import streamlit as st
from spacytextblob.spacytextblob import SpacyTextBlob


def ner(input):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(input)
    labels = [label for label in nlp.get_pipe('ner').labels]
    spacy_streamlit.visualize_ner(doc, labels=labels)

def sentiment_analysis(input):
    classification = pipeline('sentiment-analysis')
    if classification(input)[0]['label'] == 'POSITIVE':
        st.success('The Text Provided is **Positive**')
    else:
        st.error('The Text Provided is **Negative**')
    

def summarize(input):
    summarizer = pipeline("summarization", model='Falconsai/text_summarization')
    summary = summarizer(input, max_length=50, do_sample=False)
    st.success(summary[0]['summary_text'])


def translate(input):
    checkpoint = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([input], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    st.success(translated_text[0])