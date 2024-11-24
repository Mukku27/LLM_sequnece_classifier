import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from typing import List, Optional, Dict

import requests
from bs4 import BeautifulSoup, PageElement
from tqdm import tqdm

from src.definitions.enums import TypeOfWork
from src.definitions.templates import JobOpportunity, JobSeeker
from src.services.chat_model import ChatModel

logger = logging.getLogger(__name__)


class OLJScraper:
    def __init__(self, chat_model: ChatModel, job_seeker: JobSeeker, search_keywords: Optional[List[str]]):
        self.chat_model = chat_model
        self.job_seeker = job_seeker
        self.base_url = 'https://www.onlinejobs.ph'
        self.max_pages = 2
        self.max_jobs_to_process = 50
        self.search_keywords = self.search_keywords_slug(search_keywords)
        self.urls_today = self.get_jobs_today()

    @staticmethod
    def search_keywords_slug(search_keywords: List[str]) -> List[str]:
        if search_keywords is None:
            return []
        return [search_keyword.strip().replace(' ', '+') for search_keyword in search_keywords]

    @staticmethod
    def counter(page_number: int) -> int:
        # Based on OLJ
        return (page_number - 1) * 30

    def start(self) -> List[JobOpportunity]:
        logger.info("Getting jobs..")
        jobs = []

        def get_olj_job_recommendation(url: str) -> Optional[JobOpportunity]:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            overview = self.get_job_overview(soup)
            salary = overview['SALARY']
            type_of_work = TypeOfWork.work(overview['TYPE OF WORK'])

            job_description = self.get_job_details(soup)
            title = soup.find('h1').get_text(strip=True)
            job_opportunity = JobOpportunity(
                job_title=title,
                job_description=job_description,
                salary=salary,
                type_of_work=type_of_work,
                url=url
            )
            is_relevant = self.is_relevant(job_opportunity)
            # print(is_relevant)
            if not is_relevant:
                return None
            return job_opportunity

        to_process = self.urls_today
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(get_olj_job_recommendation, url)
                for url in tqdm(to_process, desc="Getting Job Recommendations")
            ]
            for future in tqdm(futures):
                result = future.result()
                if result:
                    jobs.append(result)
                    yield result
        logging.info(f"{len(jobs)} relevant jobs found.")

    def keywords_found(self, job_opportunity: JobOpportunity) -> bool:
        for keyword in self.search_keywords:
            match = re.search(keyword.lower(), job_opportunity.job_description.lower())
            if match:
                return True
        return False

    def skills_found(self, job_opportunity: JobOpportunity) -> bool:
        for skills in self.job_seeker.skills:
            match = re.search(skills.lower(), job_opportunity.job_description.lower())
            if match:
                return True
        return len(self.job_seeker.skills) == 0

    def is_relevant(self, job_opportunity: JobOpportunity) -> bool:
        # Send this to chat gpt
        keywords_found = self.keywords_found(job_opportunity)
        skills_found = self.skills_found(job_opportunity)
        if keywords_found or skills_found:
            chat_model_result = bool(self.chat_model.check_job_relevance(self.job_seeker, job_opportunity))
            # logging.info(f"Chat model result: {chat_model_result}")
            return chat_model_result
        logging.info(f"No skills or search keywords found in job post: {job_opportunity.job_title}")
        return False

    def get_jobs_today(self) -> List[str]:
        # Multi word search: Replace ' ' with +
        urls = []

        def fetch_urls(page):
            if len(self.search_keywords) == 0:
                url = self.base_url + f'/jobseekers/search/c/6/{self.counter(page + 1)}'
                return self.get_job_urls_in_page(url)

            urls = []
            for keyword in self.search_keywords:
                url = f'https://www.onlinejobs.ph/jobseekers/jobsearch/{self.counter(page + 1)}?jobkeyword={keyword}'
                urls += self.get_job_urls_in_page(url)
            return urls

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(fetch_urls, page) for page in tqdm(range(self.max_pages))]
            for future in futures:
                urls += future.result()
        return urls

    def get_job_urls_in_page(self, url: str) -> List[str]:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        jobs = soup.find_all('div', class_='jobpost-cat-box latest-job-post card-hover-default')
        urls = []

        def get_url_from_job(job: PageElement):
            date = self.get_date_from_job_post(job)
            # if date < dt.now().date():
            #     return None
            url = self.base_url + f"{job.find('a')['href']}"
            return url

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_url_from_job, job) for job in jobs]
            for future in futures:
                url = future.result()
                if url:
                    urls.append(url)
        return urls

    @staticmethod
    def get_date_from_job_post(job: PageElement) -> dt.date:
        date = job.find('em').get_text(strip=True)
        date_match = re.search(r'Posted on (\w+ \d{1,2}, \d{4})', date)
        date = date_match.group(1)
        date = dt.strptime(date, '%b %d, %Y').date()
        return date

    @staticmethod
    def get_job_details(soup: BeautifulSoup) -> str:
        job_description = soup.find('p', id='job-description')
        text = job_description.get_text(separator='\n', strip=True)
        return text

    @staticmethod
    def get_job_overview(soup: BeautifulSoup) -> Dict[str, str]:
        overview_card = soup.find('div', class_='card job-post shadow mb-4 mb-md-0')
        sections = overview_card.find_all('h3')
        mapping = {
            section.get_text(strip=True): section.find_next_sibling().get_text(strip=True)
            for section in sections
        }
        return mapping
