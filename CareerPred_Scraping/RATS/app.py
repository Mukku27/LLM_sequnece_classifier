import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# import streamlit as st
# import streamlit.components.v1 as com
from streamlit_option_menu import option_menu
# from streamlit.components.v1 import html


load_dotenv() # load all our environment varibles

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # api-calling

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro') #model
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file): # will read from file
    reader = pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#Prompt Template

input_prompt='''
Hey act like a skilled or very experienced ATS(Application Tracking System) with a deep understanding of tech field, software engineering, 
data science, data analyst and big data engineer. Your task is to evaluate the given job description based on the given job description. 
You must consider the job market is very competitive and you should provide the best assistance for improving the resumes. 
Assign the percentage matching based on job description and the missing keywords with high accuracy.
resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%", "MissingKeywords:[]", "Profile Summary":""}}
'''

#Streamlit App
st.title("Greenline")
# st.text("Improve your ATS")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload your Resume",type="pdf",help="Please upload the pdf")
submit=st.button("Submit")



# if submit:
#     if Uploaded_file is not None:
#         text=input_pdf_text(Uploaded_file)
#         response=get_gemini_response(input_prompt)
#         st.subheader(response)

if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        response=get_gemini_response(input_prompt)
        st.subheader(response)


with st.sidebar:
  selected= option_menu(
        menu_title=None,
        options=["Home"],
        icons=["house"],
        default_index=0,
        orientation="vertical",
    )
