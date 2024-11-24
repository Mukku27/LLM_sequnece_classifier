from dotenv import load_dotenv
from components.functions import Functions
from components.prompts import ats_resume, ats_score

import streamlit as st

def run_ats(llm, doc='', jd='', manual=False):
    load_dotenv()
    ats = Functions()
    
    submit = st.button("Percentage match")
    
    if submit:
        if doc is not None:
            with st.spinner("Calculating Score..."):
                if manual:
                    response, keywords = ats.calculate_ats_score(resume_data=doc, job_description=jd)
                    
                else:
                    response = ats.get_gemini_response(llm=llm, template=ats_score, doc=doc, input_text=jd)
                extra_response = ats.get_gemini_response(llm=llm, template=ats_resume, doc=doc, input_text=jd)
                
                st.subheader("The ATS Score is")
                st.write(response)
                if manual:
                    st.subheader("The Keywords Missing:")
                    for i, keyword in enumerate(keywords):
                        st.caption(f"{i+1}. {keyword}")
                st.write(extra_response)
        else:
            st.write("Please upload the resume")

if __name__ == "__main__":
    ats = Functions()
    run_ats(ats.model())