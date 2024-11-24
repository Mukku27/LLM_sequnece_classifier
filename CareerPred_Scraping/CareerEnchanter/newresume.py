#newresume.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import resume_update

def run_newresume(llm,doc='',jd='', role=''):
    load_dotenv()
    newresume = Functions()
    st.write("Generate New Resume.")
    submit = st.button("Generate Resume")    
    
    if submit:
        if doc is not None:
            with st.spinner("Generating..."):
                response=newresume.get_gemini_response(llm=llm,template=resume_update,doc=doc,input_text=jd + f"Role: {role}")
            st.subheader("Generated Resume")
            st.write(response)
        else:
            st.write("Please upload the resume")

if __name__ == "__main__":
    newresume = Functions()
    run_newresume(newresume.model())