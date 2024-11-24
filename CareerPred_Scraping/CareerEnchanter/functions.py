#fucntions.py
import os
import pyperclip
import streamlit as st
import speech_recognition as sr
import re
import numpy as np
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
# from convert import ExtractPDFText
import streamlit as st


class Functions():

    @staticmethod
    def get_gemini_response(llm, input_text, doc, template, info=''):
        formated_prompt = template.format(doc=doc, input_text=input_text, info=info)
        response = llm.invoke(formated_prompt)
        return response.content
        # return formated_prompt

    @staticmethod
    def copy_text(answer, copy_button=False):
        pyperclip.copy(answer)
        if copy_button:
            st.toast("Text copied to clipboard!", icon="📋")

    @staticmethod
    def record_audio():
        r = sr.Recognizer()
        with st.spinner("Recording..."):
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                with st.spinner("Say Something..."):
                    audio = r.listen(source, timeout=5)
            with st.spinner("Processing..."):
                try:
                    text = r.recognize_google(audio)
                    st.session_state['input_text'] = text
                    return text
                except sr.UnknownValueError:
                    st.write("Sorry, I could not understand what you said. Please try again or write in text box.")
                    return ""
                except sr.RequestError as e:
                    st.write(f"Could not request results; {e}")
                    return ""

    @staticmethod
    def input_state(input_text):
        if isinstance(input_text, str):
            st.session_state['input_text'] = input_text

    @staticmethod
    def calculate_ats_score(resume_data, job_description):
        # Download NLTK stopwords if not already downloaded
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')


        def preprocess_text(text):
            text = text.lower()
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word not in stop_words]
            string_text = ' '.join(filtered_text)
            text = re.sub(r'[^a-zA-Z\s]', '', string_text)
            return text

        def get_bert_embeddings(text):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings

        def calculate_cosine_similarity(embedding1, embedding2):
            sim = np.dot(embedding1[0].numpy(), embedding2[0].numpy()) / (
                np.linalg.norm(embedding1[0].numpy()) * np.linalg.norm(embedding2[0].numpy())
            )
            return sim

        resume = preprocess_text(resume_data)
        job_desc = preprocess_text(job_description)
        resume_embeddings = get_bert_embeddings(resume)
        job_desc_embeddings = get_bert_embeddings(job_desc)
        similarity_score = calculate_cosine_similarity(resume_embeddings, job_desc_embeddings)
        missing_keywords = [word for word in word_tokenize(job_desc) if word not in word_tokenize(resume)]
        if len(missing_keywords) == 0 :
            missing_keywords = ['Congratualitions, All the keywords match with your resume!!']
        return str(round(similarity_score * 100, 2)), missing_keywords

