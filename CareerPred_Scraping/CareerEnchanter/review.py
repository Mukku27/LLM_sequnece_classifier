#review.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import resume_review

def run_review(llm,doc='',jd='', role=''):
    load_dotenv()
    review = Functions()
    st.write("Get a review on how well your Resume is")
    submit = st.button("Tell Me About the Resume")
    
    if submit:
        with st.spinner("Reviewing Your Resume..."):
            if doc is not None:
                response=review.get_gemini_response(llm=llm,template=resume_review,doc=doc,input_text=jd, info=role)
                st.subheader("Resume Review: ")
                st.write(response)
            else:
                st.write("Please upload the resume")

if __name__ == "__main__":
    review = Functions()
    run_review(review.model())