#company.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import company_recommendations



def run_company(llm,doc='',jd='', role=''):
    load_dotenv()
    company = Functions()
    st.write("Suggesting new New Job Oppertunities based your resume")
    submit = st.button("Recommend New Oppertunities")
    
    
    if submit:
        if doc is not None:
            with st.spinner("Searching..."):
                response=company.get_gemini_response(llm=llm,template=company_recommendations,doc=doc,input_text=jd)
            st.subheader("The Job Roles You can Apply for:")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    company = Functions()
    run_company(company.model())