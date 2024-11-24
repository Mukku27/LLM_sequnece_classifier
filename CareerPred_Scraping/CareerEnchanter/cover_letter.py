#letter.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import cover_letter



def run_letter(llm,doc='',jd='', role=''):
    load_dotenv()
    letter = Functions()
    st.write("Generate a cover letter based on your resume and Provided Job Description.")
    submit = st.button("Generate Cover Letter")
    
    
    if submit:
        if doc is not None:
            with st.spinner("Generating..."):
                response = letter.get_gemini_response(llm=llm,template=cover_letter,doc=doc,input_text=jd, info=role)
            st.subheader("Cover Letter")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    letter = Functions()
    run_letter(letter.model())