#enhance.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import resume_enhance



def run_enhance(llm,doc='',jd='', role=''):
    load_dotenv()
    enhance = Functions()
    st.write("Providing bullet points for enhancing the given paragraph.")
    bullet = st.text_input("Paragraph for bullet points")
    submit = st.button("Provide Bullet Points")

    if submit:
        if doc is not None:
            with st.spinner("Processing..."):
                response=enhance.get_gemini_response(llm=llm,template=resume_enhance,doc=doc,input_text=jd, info=bullet)
            st.subheader("The Repsonse is")
            st.write(response)
        else:
            st.write("Please upload the resume")

    

if __name__ == "__main__":
    enhance = Functions()
    run_enhance(enhance.model())