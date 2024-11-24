#recommend.py

from dotenv import load_dotenv
from components.functions import Functions
import streamlit as st
from components.prompts import resume_recommendation, recommendation_section

def run_recommend(llm,doc='',jd='', section=False):
    load_dotenv()
    recommend = Functions()
    st.write("Recommendations To improve Your Resume.")
    submit = st.button("Recommend Me!!")
    
    if submit:
        if doc is not None:
            with st.spinner("Analyzing..."):
                if section:
                    response = recommend.get_gemini_response(llm=llm,template=recommendation_section,doc=doc,input_text=jd)
                else: 
                    response = recommend.get_gemini_response(llm=llm,template=resume_recommendation,doc=doc,input_text=jd)
            st.subheader("The Recommendations you can Apply in your Resume: ")
            st.write(response)
        else:
            st.write("Please upload the resume")

if __name__ == "__main__":
    recommend = Functions()
    run_recommend(recommend.model())