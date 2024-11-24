from src.scraper.olj_scraper import OLJScraper
from src.definitions.templates import JobSeeker
from src.definitions.enums import TypeOfWork
import re
import logging
from time import sleep

def test_long_function():
    for i in range(10):
        yield i
        sleep(i)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    job_seeker = JobSeeker(
        current_position="Python Developer",
        industry="Software",
        years_of_experience="2",
        skills=["Python", "AI", "Backend Development", "Flask"],
        profile="I am a Python developer specializing in AI",
        salary_expectation="Php 100000 per month",
        type_of_work=TypeOfWork.ANY
    )
    scraper = OLJScraper(job_seeker, search_keywords=["python"])
    jobs = scraper.start()
    for job in jobs:
        print(job)