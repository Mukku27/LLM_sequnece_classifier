#linkedin.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import linkedin_profile



def run_linkedin(llm,doc='',jd='', role=''):
    load_dotenv()
    linkedin = Functions()
    st.write("Suggesting Comprehensive LinkedIn Profile.")
    submit = st.button("Suggest Profile Details")
    
    
    if submit:
        if doc is not None:
            with st.spinner("Creating..."):
                response=linkedin.get_gemini_response(llm=llm,template=linkedin_profile,doc=doc,input_text=jd)
            st.subheader("Comprehensive LinkedIn Profile")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    linkedin = Functions()
    run_linkedin(linkedin.model())