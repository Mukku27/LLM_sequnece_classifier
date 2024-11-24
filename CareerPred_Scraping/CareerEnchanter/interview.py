#interview.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import interview as  interview_prompt



def run_interview(llm,doc='',jd='', role=''):
    load_dotenv()
    interview = Functions()
    st.write("Generate Customized Interview Questions.")
    submit = st.button("Generate Interview Questions")
    
    
    if submit:
        if doc is not None:
            with st.spinner("Generating..."):
                response = interview.get_gemini_response(llm=llm,template=interview_prompt,doc=doc,input_text=jd,info=role)
            st.subheader("Customized Interview Questions")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    interview = Functions()
    run_interview(interview.model())