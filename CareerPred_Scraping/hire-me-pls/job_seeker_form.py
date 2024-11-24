import logging
import threading
import requests.exceptions
import streamlit as st
from streamlit_tags import st_tags

from src.definitions.enums import Currencies, SalaryFrequency, TypeOfWork
from src.definitions.templates import JobSeeker, JobOpportunity
from src.scraper.olj_scraper import OLJScraper
from src.services.chat_model import ChatModel
from src.utils.decorators import job_seeker_info, reset

logger = logging.getLogger(__name__)


class JobSeekerForm:
    def __init__(self):

        self.chat_model = ChatModel()
        self.keys = [
            "current_position", "industry", "years_of_experience", "salary_expectation", "type_of_work", "profile", "salary_value"
        ]
        self.reset_job_info()

    def main(self):
        if st.session_state.current_position == "":
            self.current_position_screen()
            return
        elif st.session_state.industry == "":
            self.industry_screen()
            return
        elif st.session_state.years_of_experience == "":
            self.yoe_screen()
            return
        elif st.session_state.salary_value == "" or not st.session_state.salary_complete:
            self.salary_screen()
            return
        elif st.session_state.type_of_work == "" or not st.session_state.type_of_work_complete:
            self.type_of_work_screen()
            return
        elif st.session_state.profile == "":
            self.profile_screen()
            return
        elif len(st.session_state.skills) == 0 or not st.session_state.skills_complete:
            self.skills_screen()
            return
        elif not st.session_state.keywords_complete:
            self.keywords_screen()
            return
        else:
            if st.session_state.show_cover_letter:
                self.show_cover_letter()
                return
            job_seeker_attributes = {
                key: st.session_state[key]
                for key, _ in JobSeeker.__annotations__.items()
            }
            job_seeker = JobSeeker(**job_seeker_attributes)
            st.session_state.job_seeker = job_seeker
            scraper = OLJScraper(
                chat_model=self.chat_model,
                job_seeker=job_seeker,
                search_keywords=st.session_state.keywords)
            with st.spinner("Please wait"):
                try:
                    if not st.session_state.loaded_jobs and len(st.session_state.jobs_list) == 0:
                        st.session_state.jobs = scraper.start()
                        st.session_state.loaded_jobs = True
                    self.jobs_screen()
                    # jobs_thread = threading.Thread(target=self.jobs_screen)
                    # jobs_thread.start()
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error. Please try again later.")
                except Exception as e:
                    st.error(f"Something went wrong.. Error: {e}")

    """ 
    Screens 
    """
    @job_seeker_info(title="What's your current position?", reset_keys=["current_position"])
    def current_position_screen(self):
        st.session_state.current_position = st.text_input("Current Position")

    @job_seeker_info(title="What industry are you in?", reset_keys=["industry", "current_position"])
    def industry_screen(self):
        st.session_state.industry = st.text_input("Industry")

    @job_seeker_info(title="How many years of experience do you have?", reset_keys=["years_of_experience", "industry"])
    def yoe_screen(self):
        st.session_state.years_of_experience = st.text_input("Years of Experience")

    @job_seeker_info(title="What's your expected salary?",
                     reset_keys=["salary_expectation", "salary_value", "years_of_experience"],
                     callback=lambda: setattr(st.session_state, 'salary_complete', True))
    def salary_screen(self):
        currency, salary, frequency = st.columns([1, 3, 1])
        currency_value = currency.selectbox("Currency", Currencies.currencies())
        salary_value = salary.text_input("What's your expected salary?")
        frequency_value = frequency.selectbox("Frequency", SalaryFrequency.frequencies())
        st.session_state.salary_value = salary_value
        st.session_state.salary_expectation = f'{currency_value} {salary_value} - {frequency_value}'

    @job_seeker_info(title="What type of work are you looking for?",
                     reset_keys=['type_of_work', 'salary_expectation', 'salary_value'],
                     callback=lambda: setattr(st.session_state, 'type_of_work_complete', True))
    def type_of_work_screen(self):
        st.session_state.type_of_work = st.selectbox("Type of Work", TypeOfWork.types())

    @job_seeker_info(title="Tell me more about yourself", reset_keys=['profile', 'type_of_work'])
    def profile_screen(self):
        st.session_state.profile = st.text_area("Profile", height=300)

    @job_seeker_info(title="What are you good at?",
                     reset_keys=['skills', 'profile'],
                     callback=lambda: setattr(st.session_state, 'skills_complete', True))
    def skills_screen(self):
        st.session_state.skills = st_tags(label="Skills", text="")

    @job_seeker_info(title="Anything specific you're looking for?",
                     reset_keys=['keywords', 'skills'],
                     callback=lambda: setattr(st.session_state, 'keywords_complete', True))
    def keywords_screen(self):
        st.session_state.keywords = st_tags(label="Search Keywords", text="")

    def jobs_screen(self):
        st.title("Jobs")
        st.button("Go Back", on_click=self.clear_jobs)
        st.session_state.is_loading = True
        if len(st.session_state.jobs_list) == 0:
            for job in st.session_state.jobs:
                st.session_state.jobs_list.append(job)
                with st.expander(job.job_title):
                    self.show_project_details(job)
        else:
            # Generate from jobs list
            for job in st.session_state.jobs_list:
                with st.expander(job.job_title):
                    self.show_project_details(job)
        st.session_state.is_loading = False

    def show_cover_letter(self):
        st.title("Cover Letter")
        st.button("Back", on_click=lambda: reset(["cover_letter", "show_cover_letter"]))
        st.write(st.session_state.cover_letter)

    """
    Helpers
    """
    def reset_job_info(self):
        if "start_search" not in st.session_state:
            st.session_state.start_search = False
        if "skills_complete" not in st.session_state:
            st.session_state.skills_complete = False
        if "keywords_complete" not in st.session_state:
            st.session_state.keywords_complete = False
        if "salary_complete" not in st.session_state:
            st.session_state.salary_complete = False
        if "type_of_work_complete" not in st.session_state:
            st.session_state.type_of_work_complete = False
        if "skills" not in st.session_state:
            st.session_state.skills = []
        if "keywords" not in st.session_state:
            st.session_state.keywords = []
        if "job_seeker" not in st.session_state:
            st.session_state.job_seeker = ""
        if "loaded_jobs" not in st.session_state:
            st.session_state.loaded_jobs = False
        if "jobs_list" not in st.session_state:
            st.session_state.jobs_list = []
        if "show_cover_letter" not in st.session_state:
            st.session_state.show_cover_letter = False
        if "cover_letter" not in st.session_state:
            st.session_state.cover_letter = ""
        if "is_loading" not in st.session_state:
            st.session_state.is_loading = False

        for key in self.keys:
            if key not in st.session_state:
                st.session_state[key] = ""

    def clear_jobs(self):
        st.session_state.jobs = []
        st.session_state.skills = []
        st.session_state.jobs = []
        for key in self.keys:
            st.session_state[key] = ""

    def show_project_details(self, job: JobOpportunity):
        try:
            st.write(f'### {job.job_title}')
            st.markdown(f"[Read more]({job.url})")
            st.write(f"Salary: {job.salary}")
            st.write(f"Type of Work: {job.type_of_work.value}")
            # st.button("Generate Cover Letter",
            #           key=job.job_description,
            #           on_click=self.on_click_generate_cover_letter,
            #           args=(st.session_state.job_seeker, job))
        except Exception as e:
            logger.error(f"Error showing project details. Error: {e}")

    def on_click_generate_cover_letter(self, job_seeker: JobSeeker, job_opportunity: JobOpportunity):
        # This is kinda buggy
        st.session_state.cover_letter = self.chat_model.generate_cover_letter(job_seeker=job_seeker, job_opportunity=job_opportunity)
        st.session_state.show_cover_letter = True

