# app.py
import os
import streamlit as st
from features import (ats, 
                      analyzer, 
                      company_recommend, 
                      cover_letter, 
                      enhance, 
                      improve, 
                      interview,
                      linkedin,
                      newresume,
                      recommend, 
                      review)
from components import docLoader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize CareerEnchanter
class CareerEnchanter(object):
    def __init__(self, title="CareerEnchanter"):
        self.title = title

    @staticmethod
    def model():
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return ChatGoogleGenerativeAI(model="gemini-pro")

# Initialize CareerEnchanter instance
enchanter = CareerEnchanter()

# Set Streamlit page configuration
st.set_page_config(page_title=enchanter.title, page_icon='🤖', layout='wide')

# Main title
st.title("🚀 Career Enchanter 🚀")

# Load document
text = docLoader.load_doc()
st.session_state['doc_text'] = text

jd, doc = st.columns(2)
with jd:
    # Job Description input
    jd = st.text_area("Job Description: ", key="input")
if text: 
    with doc:
        extracted= st.text_area("Extracted Data From Resume", value=st.session_state['doc_text'])

role=st.text_input("Role you want to Apply for")
st.session_state['role'] = role

# Sidebar options
with st.sidebar:
    st.title('🔮 Career Enchanter 🔮')
    st.subheader('Options: ')
    option = st.radio("Select an option: ", (
        "ATS Score", 
        "Resume Review", 
        "Resume Enhancements", 
        "Resume Improvements", 
        "Recommendation", 
        "Keywords", 
        "Generate Cover Letter", 
        "Resume Generator", 
        "Linkedin Profile Update", 
        "Possible Interview Questions", 
        "Company Recommendations"
        ))

    # Load model
    with st.spinner("Loading Model..."):
        llm = enchanter.model()
if option == "ATS Score":
            calculation_method = st.radio("Choose how you want to calculate ATS Score: ", ("Using AI", "Manually (Cosine Similarity)"), horizontal=True)

elif option == "Recommendation":
            recommendation_type = st.radio("Select the type of recommendation you want: ", ("Entire Resume", "Section Wise"), horizontal=True)

elif option == "Keywords":
            analyz_type = st.radio("Select the type of Keywords Fucntion you want: ", ("Analyse Keywords", "Keyword Synonyms"), horizontal=True)
# Dictionary mapping options to functions
option_functions = {
    "ATS Score": ats.run_ats,
    "Resume Review": review.run_review,
    "Resume Enhancements": enhance.run_enhance,
    "Resume Improvements": improve.run_improve,
    "Recommendation": recommend.run_recommend,
    "Keywords": analyzer.run_analyzer,
    "Generate Cover Letter": cover_letter.run_letter,
    "Resume Generator": newresume.run_newresume,
    "Linkedin Profile Update": linkedin.run_linkedin,
    "Possible Interview Questions": interview.run_interview,
    "Company Recommendations": company_recommend.run_company
}

# Handle the selected option
if option in option_functions:
    func = option_functions[option]
    if option == "ATS Score":
        if calculation_method == "Manually (Cosine Similarity)":
            func(llm, st.session_state['doc_text'], jd, manual=True)
        else:
            func(llm, st.session_state['doc_text'], jd)
    elif option == "Recommendation":
        if recommendation_type == "Entire Resume":
            func(llm, st.session_state['doc_text'], jd, section=True)
        else:
            func(llm, st.session_state['doc_text'], jd)
    elif option == "Keywords":
        if analyz_type == "Analyse Keywords":
            func(llm, st.session_state['doc_text'], jd, analysis=True)
        else:
            func(llm, st.session_state['doc_text'], jd)
    else:
        func(llm, st.session_state['doc_text'], jd, role=st.session_state['role'])
