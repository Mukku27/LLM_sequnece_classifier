import logging

import streamlit as st

from src.app.job_seeker_form import JobSeekerForm
from src.definitions.templates import TypeOfWork, PROFILE

logger = logging.getLogger(__name__)


class App:
    def __init__(self):
        self.job_seeker_form = JobSeekerForm()
        self.keys = [
            "current_position", "industry", "years_of_experience", "salary", "type_of_work", "profile"
        ]
        self._initialize_session_state()
        self._initialize_app()

    @staticmethod
    def _initialize_app():
        st.set_page_config(page_title="hire-me-pls", layout='centered')

    @staticmethod
    def _initialize_session_state():
        if "page" not in st.session_state:
            st.session_state.page = "main"
        if "loading" not in st.session_state:
            st.session_state.loading = False
        if "jobs" not in st.session_state:
            st.session_state.jobs = []
        if "salary_value" not in st.session_state:
            st.session_state.salary_value = ""

    def validate_inputs(self):
        for key in self.keys:
            if st.session_state[key] == "":
                st.error("Empty fields found.")
                return False
        return True

    def main(self):
        # self.generate_test()
        self.job_seeker_form.main()

    def generate_test(self):
        st.session_state.current_position = "AI Engineer"
        st.session_state.industry = "Software"
        st.session_state.years_of_experience = "2"
        st.session_state.salary_value = "10"
        st.session_state.salary_complete = True
        st.session_state.type_of_work_complete = True
        st.session_state.type_of_work = "Any"
        st.session_state.profile = PROFILE
        st.session_state.skills = ["Python", "OpenAI", "Flask", "FastAPI", "BeautifulSoup", "Redis", "Streamlit"]
        st.session_state.skills_complete = True
        st.session_state.keywords = ["python"]
        st.session_state.keywords_complete = True
