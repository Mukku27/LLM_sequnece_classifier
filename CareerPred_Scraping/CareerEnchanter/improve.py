#improve.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import resume_improve



def run_improve(llm,doc='',jd='', role=''):
    load_dotenv()
    improve = Functions()
    st.write("Suggesting new elements to add to your resume based on the existing content.")
    submit = st.button("How Can I Improvise my Skills")
    
    
    if submit:
        if doc is not None:
            with st.spinner("Generating..."):
                response=improve.get_gemini_response(llm=llm,template=resume_improve,doc=doc,input_text=jd)
            st.subheader("The Required Improvmemts are")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    improve = Functions()
    run_improve(improve.model())