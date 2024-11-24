from dotenv import load_dotenv
from components.functions import Functions
from components.prompts import keyword_analysis, keyword_synonyms

import streamlit as st

def run_analyzer(llm, doc='', jd='', analysis=False):
    load_dotenv()
    analyzer = Functions()
    keyword = ''
    if analysis:
        message = "Suggesting Keywords to add in your Resume."
        template = keyword_analysis
    else:
        message = "Suggesting Synonyms for Provided Keywords to add in your Resume."
        keyword = st.text_input("Keyword")
        if not keyword:
            st.write("Please provide a keyword for synonyms.")
            return
        template = keyword_synonyms

    st.write(message)
    submit = st.button("Suggest Keywords" if analysis else "Generate Keywords")
    
    if submit:
        if doc is not None:
            with st.spinner("Analyzing..."):
                response = analyzer.get_gemini_response(llm=llm, template=template, doc=doc, input_text=jd, info=keyword)
                st.subheader("The Keywords You Can Add:")
                st.write(response)
        else:
            st.write("Please upload the resume")

if __name__ == "__main__":
    analyzer = Functions()
    run_analyzer(analyzer.model())